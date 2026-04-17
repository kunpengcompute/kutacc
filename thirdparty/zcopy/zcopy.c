#include <linux/init.h>
#include <linux/module.h>
#include <linux/kallsyms.h>
#include <linux/mm.h>
#include <linux/ioctl.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>
#include <linux/version.h>
#include <linux/kernel.h>
#include <linux/kprobes.h>
#include <linux/mm_types.h>
#include <linux/list.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/huge_mm.h>
#include <linux/mm_types.h>
#include <linux/mm_types_task.h>
#include <linux/rmap.h>
#include <asm-generic/pgalloc.h>
#include <asm/tlbflush.h>
#include <asm/pgtable.h>
#include <asm/pgalloc.h>
#include <asm/pgtable-hwdef.h>

/* 局部宏定义 */


#define EN_DEBUG    1                     /* 调试信息开关 */
#if EN_DEBUG
#define PRINT(x...) printk(KERN_EMERG x)  /* 提高打印等级 */   
#else
#define PRINT(x...)
#endif

#ifndef PUD_SHIFT
#define ARM64_HW_PGTABLE_LEVEL_SHIFT(n) ((PAGE_SHIFT - 3) * (4 - (n)) + 3)
#define PUD_SHIFT ARM64_HW_PGTABLE_LEVEL_SHIFT(1)
#endif

/* 局部变量 */


typedef enum {
    IO_NONE     = 0,
    IO_ATTACH   = 1,
    IO_DUMP     = 3,
    IO_MAX
} IOCTL_NUMBER;

struct dax_ioctl_pswap {
    unsigned long src_addr;
    unsigned long dst_addr;
    int src_pid;
    int dst_pid;
    unsigned long size;
};
typedef struct dax_ioctl_pswap dax_ioctl_pswap_t;

struct dax_ioctl_mmap {
    unsigned long size;
    unsigned long addr;
};
typedef struct dax_ioctl_mmap dax_ioctl_mmap_t;

struct sls_cdev {
    struct cdev    chrdev;
    dev_t          dev;
    int            major;
    struct class  *dev_class;
    struct device *dev_device;
};

static DEFINE_MUTEX(attach_lock);

static struct sls_cdev s_cdev;

struct mm_struct *sls_init_mm = NULL;

int (*__sls_pte_alloc)(struct mm_struct *mm, pmd_t *pmd);
int (*__sls_pmd_alloc)(struct mm_struct *mm, pud_t *pud, unsigned long address);
int (*__sls_pud_alloc)(struct mm_struct *mm, p4d_t *p4d, unsigned long address);
//int (*__sls_p4d_alloc)(struct mm_struct *mm, pgd_t *pgd, unsigned long address);

#define REGISTER_CHECK(_var, _errstr)               \
    do {                                            \
        if (!_var) {                                \
            printk("Not fount %s\n", _errstr);      \
            return -1;                              \
        }                                           \
    } while(0)

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 7, 0)
static unsigned long (*kallsyms_lookup_name_funcp)(const char *symbol_name);

static int entry_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
    return 0;
}

static int ret_handler(struct kretprobe_instance *ri, struct pt_regs *regs)
{
    return 0;
}

static struct kretprobe __kretprobe = {
    .handler = ret_handler,
    .entry_handler = entry_handler,
};

static unsigned long __kprobe_lookup_name(const char *symbol_name)
{
    int ret;
    void *addr;

    __kretprobe.kp.symbol_name = symbol_name;
    ret = register_kretprobe(&__kretprobe);
    if(ret < 0) {
        pr_err("register_kprobe failed, returned %d\n", ret);
        return 0;
    }
    pr_info("Planted %s kprobe at %pK\n", symbol_name, __kretprobe.kp.addr);
    addr = __kretprobe.kp.addr;
    unregister_kretprobe(&__kretprobe);
    return (unsigned long)addr;
}

static inline unsigned long __kallsyms_lookup_name(const char *symbol_name)
{
    if (kallsyms_lookup_name_funcp == NULL)
        return 0;
    return kallsyms_lookup_name_funcp(symbol_name);
}
#else
static inline unsigned long __kallsyms_lookup_name(const char *symbol_name)
{
    return kallsyms_lookup_name(symbol_name);
}
#endif

static inline pud_t *sls_pud_alloc(struct mm_struct *mm, p4d_t *p4d,
                                     unsigned long address)
{
    return (unlikely(p4d_none(*p4d)) && __sls_pud_alloc(mm, p4d, address)) ?
            NULL : pud_offset(p4d, address);
}

static inline pmd_t *sls_pmd_alloc(struct mm_struct *mm, pud_t *pud, unsigned long address)
{
    return (unlikely(pud_none(*pud)) && __sls_pmd_alloc(mm, pud, address)) ?
            NULL : pmd_offset(pud, address);
}

#define sls_pte_alloc(mm, pmd)                          \
        (unlikely(pmd_none(*(pmd))) && __sls_pte_alloc(mm, pmd))

#define sls_pte_alloc_map(mm, pmd, address)              \
        (sls_pte_alloc(mm, pmd) ? NULL : pte_offset_map(pmd, address))

/*
 *  Dump out the page tables associated with 'addr' in the currently active mm.
 */
void dump_pagetable(unsigned long addr)
{
    struct mm_struct *mm;
    pgd_t *pgdp;
    pgd_t pgd;

    if (is_ttbr0_addr(addr)) {
        /* TTBR0 */
        mm = current->active_mm;
        if (mm == sls_init_mm) {
            pr_alert("[%016lx] user address but active_mm is swapper\n",
                 addr);
            return;
        }
    } else if (is_ttbr1_addr(addr)) {
        /* TTBR1 */
        mm = sls_init_mm;
    } else {
        pr_alert("[%016lx] address between user and kernel address ranges\n",
                 addr);
        return;
    }

    pr_alert("%s pgtable: %luk pages, %u-bit VAs, pgdp = %p\n",
             mm == sls_init_mm ? "swapper" : "user", PAGE_SIZE / SZ_1K,
             VA_BITS, mm->pgd);
    pgdp = pgd_offset(mm, addr);
    pgd = READ_ONCE(*pgdp);
    pr_alert("[%016lx] pgd=%016llx", addr, pgd_val(pgd));

    do {
        p4d_t *p4dp, p4d;
        pud_t *pudp, pud;
        pmd_t *pmdp, pmd;
        pte_t *ptep, pte;

        if (pgd_none(pgd) || pgd_bad(pgd))
            break;

        p4dp = p4d_offset(pgdp, addr);
        p4d = READ_ONCE(*p4dp);
        pr_cont(", p4d=%016llx", p4d_val(p4d));
        if (p4d_none(p4d) || p4d_bad(p4d))
            break;

        pudp = pud_offset(p4dp, addr);
        pud = READ_ONCE(*pudp);
        pr_cont(", pud=%016llx", pud_val(pud));
        if (pud_none(pud) || pud_bad(pud))
            break;

        pmdp = pmd_offset(pudp, addr);
        pmd = READ_ONCE(*pmdp);
        pr_cont(", pmd=%016llx", pmd_val(pmd));
        if (pmd_none(pmd) || pmd_bad(pmd))
            break;

        ptep = pte_offset_map(pmdp, addr);
        pte = READ_ONCE(*ptep);
        pr_cont(", pte=%016llx, pfn=:%016llx", pte_val(pte), pte_pfn(pte));
        pte_unmap(ptep);
    } while(0);

    pr_cont("\n");
}

static pud_t *get_old_pud(struct mm_struct *mm, unsigned long addr)
{
    pgd_t *pgd;
    p4d_t *p4d;
    pud_t *pud;

    pgd = pgd_offset(mm, addr);
    if (pgd_none(*pgd))
        return NULL;

    p4d = p4d_offset(pgd, addr);
    if (p4d_none(*p4d))
        return NULL;

    pud = pud_offset(p4d, addr);
    if (pud_none(*pud))
        return NULL;

    return pud;
}

static pmd_t *get_old_pmd(struct mm_struct *mm, unsigned long addr)
{
    pud_t *pud;
    pmd_t *pmd;

    pud = get_old_pud(mm, addr);
    if (!pud)
        return NULL;

    pmd = pmd_offset(pud, addr);
    if (pmd_none(*pmd))
        return NULL;

    return pmd;
}

static pud_t *alloc_new_pud(struct mm_struct *mm, unsigned long addr)
{
    pgd_t *pgd;
    p4d_t *p4d;

    pgd = pgd_offset(mm, addr);
    p4d = p4d_alloc(mm, pgd, addr);
    if (!p4d)
        return NULL;

    return sls_pud_alloc(mm, p4d, addr);
}

static pmd_t *alloc_new_pmd(struct mm_struct *mm, unsigned long addr)
{
    pud_t *pud;
    pmd_t *pmd;

    pud = alloc_new_pud(mm, addr);
    if (!pud)
        return NULL;

    pmd = sls_pmd_alloc(mm, pud, addr);
    if (!pmd)
        return NULL;

    VM_BUG_ON(pmd_trans_huge(*pmd));

    return pmd;
}

enum pgt_entry {
    NORMAL_PMD,
    HPAGE_PMD,
    NORMAL_PUD,
    HPAGE_PUD,
};

static __always_inline unsigned long get_extent(enum pgt_entry entry,
            unsigned long old_addr, unsigned long old_end,
            unsigned long new_addr)
{
    unsigned long next, extent, mask, size;

    switch (entry) {
    case HPAGE_PMD:
    case NORMAL_PMD:
        mask = PMD_MASK;
        size = PMD_SIZE;
        break;
    default:
        BUILD_BUG();
        break;
    }

    next = (old_addr + size) & mask;
    /* even if next overflowed, extent below will be ok */
    extent = next - old_addr;
    if (extent > old_end - old_addr)
        extent = old_end - old_addr;
    pr_info("old_addr=%lx, next=%lx, size=%ld, extent=%ld\n", old_addr, next, size, extent);
    next = (new_addr + size) & mask;
    if (extent > next - new_addr)
        extent = next - new_addr;
    pr_info("new_addr=%lx, next=%lx, size=%ld, extent=%ld\n", new_addr, next, size, extent);
    return extent;
}

static void attach_ptes(struct vm_area_struct *src_vma, pmd_t *src_pmdp,
        unsigned long src_addr, unsigned long src_addr_end,
        struct vm_area_struct *dst_vma, pmd_t *dst_pmdp,
        unsigned long dst_addr)
{
    struct mm_struct *dst_mm = dst_vma->vm_mm;
    struct mm_struct *src_mm = src_vma->vm_mm;
    pte_t *src_ptep, *dst_ptep, pte, orig_pte;
    struct page *src_page, *orig_page;
    spinlock_t *src_ptl, *dst_ptl;
    unsigned long len = src_addr_end - src_addr;

    pr_info("task[%lx] begin remap ptes from:[%lx, %lx] to:[%lx,..]\n", (unsigned long)current, src_addr, src_addr_end, dst_addr);
    src_ptep = pte_offset_map(src_pmdp, src_addr);
    dst_ptep = pte_offset_map(dst_pmdp, dst_addr);
    dst_ptl = pte_lockptr(dst_mm, dst_pmdp);
    spin_lock_nested(dst_ptl, SINGLE_DEPTH_NESTING);

    for (; src_addr < src_addr_end; src_ptep++, src_addr += PAGE_SIZE, 
                   dst_ptep++, dst_addr += PAGE_SIZE) {
        /*
         * For special pte, there may not be corresponding page. Hence,
         * we skip this situation.
         */
        if (pte_none(*src_ptep) || pte_special(*src_ptep))
            continue;
        pte = *src_ptep;
        src_page = pte_page(pte);
        atomic_inc(&src_page->_refcount);
        atomic_inc(&src_page->_mapcount);

        /* If dst virtual addr has page mapping, before setup the new mapping. 
         * we should decrease the orig page mapcount and refcount. */
        orig_pte = *dst_ptep;
        if(!pte_none(orig_pte)) {
            orig_page = pte_page(orig_pte);
            atomic_dec(&orig_page->_refcount);
            atomic_dec(&orig_page->_mapcount);
        }
        set_pte_at(dst_mm, dst_addr, dst_ptep, pte);
    }

    spin_unlock(dst_ptl);
    flush_tlb_range(dst_vma, dst_addr, dst_addr + len);
    pr_info("task[%lx] finished remap ptes from:[%lx, %lx] to:[%lx,..]\n",
        (unsigned long)current, src_addr, src_addr_end, dst_addr);
}

spinlock_t *__pmd_trans_huge_lock(pmd_t *pmd, struct vm_area_struct *vma)
{
    spinlock_t *ptl;
    ptl = pmd_lock(vma->vm_mm, pmd);
    if (likely(is_swap_pmd(*pmd) || pmd_trans_huge(*pmd) ||
            pmd_devmap(*pmd)))
        return ptl;
    spin_unlock(ptl);
    return NULL;
}

void pgtable_trans_huge_deposit(struct mm_struct *mm, pmd_t *pmdp,
                pgtable_t pgtable)
{
    assert_spin_locked(pmd_lockptr(mm, pmdp));

    /* FIFO */
    if (!pmd_huge_pte(mm, pmdp))
        INIT_LIST_HEAD(&pgtable->lru);
    else
        list_add(&pgtable->lru, &pmd_huge_pte(mm, pmdp)->lru);
    pmd_huge_pte(mm, pmdp) = pgtable;
}

bool attach_huge_pmd(struct vm_area_struct *src_vma, struct vm_area_struct *dst_vma,
        unsigned long src_addr, unsigned long dst_addr, pmd_t *old_pmdp, pmd_t *new_pmdp)
{
    spinlock_t *src_ptl, *dst_ptl;
    pmd_t pmd, orig_pmd;
    struct page *src_thp_page, *orig_thp_page;
    struct mm_struct *dst_mm = NULL;
    struct mm_struct *src_mm = NULL;
    pgtable_t pgtable = NULL;

    if (!vma_is_anonymous(dst_vma) || !vma_is_anonymous(src_vma)) {
        pr_err("dst_vma is %d and src_vma is %d\n",
            vma_is_anonymous(dst_vma), vma_is_anonymous(src_vma));
        return false;
    }

    dst_mm = dst_vma->vm_mm;
    src_mm = src_vma->vm_mm;

    // new pgtable to add new pmdp
    pgtable = pte_alloc_one(dst_mm);
    if (unlikely(!pgtable)) {
        pr_err("pte_alloc_one failed\n");
        return false;
    }

    pr_info("Begin: remap huge pmd from:%lx-%lx to:%lx-%lx\n",
            (unsigned long)src_mm, src_addr, (unsigned long)dst_mm, dst_addr);

    src_ptl = pmd_lockptr(src_mm, old_pmdp);
    dst_ptl = pmd_lockptr(dst_mm, new_pmdp);
    pr_info("src_ptl=%lx, dst_ptl=%lx\n", src_ptl, dst_ptl);

    spin_lock(src_ptl);
    pmd = *old_pmdp;
    src_thp_page = pmd_page(pmd);
    VM_BUG_ON_PAGE(!PageHead(src_thp_page), src_thp_page);
    get_page(src_thp_page);
    atomic_inc(compound_mapcount_ptr(src_thp_page));
    spin_unlock(src_ptl);

    spin_lock_nested(dst_ptl, SINGLE_DEPTH_NESTING);
    orig_pmd = *new_pmdp;
    if(!pmd_none(orig_pmd)) { /* umap the old pages */
        orig_thp_page = pmd_page(orig_pmd);
        put_page(orig_thp_page);
        atomic_dec(compound_mapcount_ptr(orig_thp_page));
    }
    pgtable_trans_huge_deposit(dst_mm, new_pmdp, pgtable);
    set_pmd_at(dst_mm, dst_addr, new_pmdp, pmd);
    spin_unlock(dst_ptl);
    flush_tlb_range(dst_vma, dst_addr, dst_addr + HPAGE_PMD_SIZE);

    pr_info("End: remap huge pmd from:%lx-%lx to:%lx-%lx\n",
            (unsigned long)src_mm, src_addr, (unsigned long)dst_mm, dst_addr);

    return true;
}

int attach_page_range(unsigned long dst_addr, unsigned long src_addr,
                    struct mm_struct *dst_mm, struct mm_struct *src_mm, unsigned long size)
{
    unsigned long extent, src_addr_end;
    unsigned dst_addr_begin;
    pmd_t *old_pmd, *new_pmd;
    int ret = 0;
    struct vm_area_struct *src_vma, *dst_vma;

    src_addr_end = src_addr + size;
    dst_addr_begin = dst_addr;
    src_vma = find_vma(src_mm, src_addr);
    dst_vma = find_vma(dst_mm, dst_addr);
    for (; src_addr < src_addr_end; src_addr += extent, dst_addr += extent) {
        cond_resched();

        // skip remapping for pud, make case simple
        extent = get_extent(NORMAL_PMD, src_addr, src_addr_end, dst_addr);
        old_pmd = get_old_pmd(src_mm, src_addr);
        if (!old_pmd)
            continue;
        new_pmd = alloc_new_pmd(dst_mm, dst_addr);
        if (!new_pmd)
            break;
        if (is_swap_pmd(*old_pmd) || pmd_trans_huge(*old_pmd) ||
            pmd_devmap(*old_pmd)) {
                if (extent == HPAGE_PMD_SIZE
                        && attach_huge_pmd(src_vma, dst_vma, src_addr, dst_addr, old_pmd, new_pmd))
                    continue;
            }
        
        if (sls_pte_alloc(dst_mm, new_pmd))
            break;
        attach_ptes(src_vma, old_pmd, src_addr, src_addr + extent, dst_vma,
                new_pmd, dst_addr);
    }
    //batch flush, reduce time usage
    flush_tlb_range(dst_vma, dst_addr_begin, dst_addr_begin + size);
    return ret;
}

int attach_pages(unsigned long dst_addr, unsigned long src_addr,
                    int dst_pid, int src_pid, unsigned long size)
{
    int ret = 0;
    struct mm_struct *dst_mm, *src_mm;
    struct task_struct *src_task, *dst_task;

    //src_task = find_task_by_pid_ns(src_pid, &init_pid_ns);
    src_task = find_get_task_by_vpid(src_pid);
    if(src_task == NULL) {
        return -EINVAL;
    }
    //dst_task = find_task_by_pid_ns(dst_pid, &init_pid_ns);
    dst_task = find_get_task_by_vpid(dst_pid);
    if(dst_task == NULL) {
        return -EINVAL;
    }

    src_mm = src_task->mm;
    dst_mm = dst_task->mm;

    if(src_mm == dst_mm) {
        return -EINVAL;
    }
    if(size <= 0) {
        return -EINVAL;
    }
    //check the addr is in userspace.
    if (!is_ttbr0_addr(dst_addr) || !is_ttbr0_addr(src_addr)) {
        return -EINVAL;
    }
    if(!src_mm || !dst_mm) {
        pr_err("task exit, src_mm=%lx, dst_mm=%lx\n",
                (unsigned long)src_mm, (unsigned long)dst_mm);
        return -EINVAL;
    }

    ret = attach_page_range(dst_addr, src_addr, dst_mm, src_mm, size);

    return ret;
}

int zcopy_open(struct inode *inode, struct file *fp)
{
    return 0;
}

int zcopy_release(struct inode *inode, struct file *fp)
{
    return 0;
}

long zcopy_ioctl(struct file *file, unsigned int type, unsigned long ptr)
{
    long ret = 0;
    printk(KERN_INFO "IOCTL begin: type:%u, cur pid:%u\n", type, current->pid);
    switch (type) {
        case IO_ATTACH:
        {
            dax_ioctl_pswap_t ctx;
            if (copy_from_user((void *)&ctx, (void *)ptr, sizeof(dax_ioctl_pswap_t))) {
                printk(KERN_ERR "copy from user for attach failed\n");
                ret = -EFAULT;
                break;
            }

            ret = attach_pages(ctx.dst_addr, ctx.src_addr, ctx.dst_pid, ctx.src_pid, ctx.size);
            break;
        }
        case IO_DUMP:
        {
            dax_ioctl_mmap_t param;
            if (copy_from_user((void *)&param, (void *)ptr, sizeof(dax_ioctl_mmap_t))) {
                printk(KERN_ERR "copy from user for dump failed\n");
                ret = -EFAULT;
                break;
            }
            dump_pagetable(param.addr);
            break;
        }
        default:
            break;
    }
    printk(KERN_INFO "IOCTL end: type:%u, cur pid:%u\n", type, current->pid);
    return ret;
}

static struct file_operations sls_fops = {
        .owner = THIS_MODULE,
        .open = zcopy_open,
        .release = zcopy_release,
        .unlocked_ioctl = zcopy_ioctl,
};

int register_unexport_func(void)
{
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 7, 0)
    kallsyms_lookup_name_funcp = (unsigned long (*)(const char *))__kprobe_lookup_name("kallsyms_lookup_name");
    pr_info("kallsyms_lookup_name_funcp = %pK\n", kallsyms_lookup_name_funcp);
#endif

    sls_init_mm = (struct mm_struct *)__kallsyms_lookup_name("init_mm");
    REGISTER_CHECK(sls_init_mm, "init_mm");

    __sls_pte_alloc = __kallsyms_lookup_name("__pte_alloc");
    REGISTER_CHECK(__sls_pte_alloc, "__pte_alloc");

    __sls_pmd_alloc = __kallsyms_lookup_name("__pmd_alloc");
    REGISTER_CHECK(__sls_pmd_alloc, "__pmd_alloc");

    __sls_pud_alloc = __kallsyms_lookup_name("__pud_alloc");
    REGISTER_CHECK(__sls_pud_alloc, "__pud_alloc");

    return 0;
}

int register_device_sls(void)
{
    int ret;
    /* register char device number */
    int error = alloc_chrdev_region(&s_cdev.dev, 0, 1, "sls");
    if ( error < 0 ) {
        printk(KERN_ERR "alloc chrdev failed\n");
        ret = -EBUSY;
        return ret;
    }

    s_cdev.major = MAJOR(s_cdev.dev);

    /* bind fs ops for char device */
    cdev_init(&s_cdev.chrdev, &sls_fops);

    /* add char device */
    error = cdev_add(&s_cdev.chrdev, s_cdev.dev, 1);
    if ( error < 0 ) {
        printk(KERN_ERR "add char device failed\n");
        ret = -EBUSY;
        return ret;
    }

    /* create class */
    s_cdev.dev_class = class_create(THIS_MODULE, "sls");
    if (IS_ERR(s_cdev.dev_class)) {
        printk("class create error\n");
        ret = -EBUSY;
        return ret;
    }

    /* create device */
    s_cdev.dev_device = device_create(s_cdev.dev_class, NULL, MKDEV(s_cdev.major, 0), NULL, "dax1.0");
    if ( NULL == s_cdev.dev_device) {
        printk("device create error\n");
        ret = -EBUSY;
        return ret;        
    }

    return 0;
}

void unregister_device_sls(void)
{
    device_destroy(s_cdev.dev_class, MKDEV(s_cdev.major, 0));
    class_destroy(s_cdev.dev_class);
    cdev_del(&s_cdev.chrdev);
    unregister_chrdev_region(s_cdev.dev, 1);
}

static int __init sls_init(void)
{
    int ret;
    PRINT("[KERNEL]:%s ------ \n",__FUNCTION__);
    ret = register_unexport_func();
    if(ret) {
        printk(KERN_ERR "register_unexport_func failed\n");
        return -1;
    }

    ret = register_device_sls();
    if(ret) {
        printk(KERN_ERR "register_device_sls failed\n");
        return -1;
    }

    return 0;
}

static void __exit sls_exit(void)
{
    unregister_device_sls();
    PRINT("[KERNEL]:%s ------ \n",__FUNCTION__);
}

module_init(sls_init);
module_exit(sls_exit);


MODULE_DESCRIPTION("SLS zero-copy module!");

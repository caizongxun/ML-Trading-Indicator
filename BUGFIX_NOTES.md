# Bug修复日志

## 问题描述

### 错误信息
```
Could not find function or function reference 'normalize_value'
```

### 根本原因
在Pine Script v5中，**函数必须在使用之前定义**。在原始代码中：
- `normalize_value()` 函数定义在 **SECTION 6** 
- 但在 **SECTION 4** 中已经调用了这个函数
- 导致编译器找不到函数定义

---

## 修复方案

### 修改内容

将 `normalize_value()` 函数定义从 **SECTION 6** 移动到 **SECTION 0**（最前面）

**Before (错误的顺序):**
```pine
// SECTION 4: 计算3个高级ML指标
lii_normalized = normalize_value(lii, 0, 100)  // ❌ 这里调用
mtri_normalized = normalize_value(mtri, 0, 100)
dsrfi_normalized = normalize_value(dsrfi, 0, 100)

// ... 很多代码

// SECTION 6: ML 训练数据准备
normalize_value(val, min_val, max_val) =>       // ❌ 但这里才定义
    if max_val == min_val
        0.5
    else
        math.max(0, math.min(1, (val - min_val) / (max_val - min_val)))
```

**After (正确的顺序):**
```pine
// SECTION 0: 辅助函数定义 (必须放在最前面)
normalize_value(val, min_val, max_val) =>       // ✓ 先定义
    if max_val == min_val
        0.5
    else
        math.max(0, math.min(1, (val - min_val) / (max_val - min_val)))

// ... 其他输入参数

// SECTION 4: 计算3个高级ML指标
lii_normalized = normalize_value(lii, 0, 100)  // ✓ 然后再调用
mtri_normalized = normalize_value(mtri, 0, 100)
dsrfi_normalized = normalize_value(dsrfi, 0, 100)
```

---

## Pine Script v5 最佳实践

### 函数定义顺序

1. **版本声明和study/indicator()**
2. **自定义函数定义** ← **必须最先**
3. **输入参数 (inputs)**
4. **指标计算**
5. **绘图和输出**

### Pine Script 函数声明语法

```pine
// v5 语法
function_name(param1, param2) =>
    expression_or_block

// 多行函数
function_name(param1) =>
    var result = 0
    result := result + param1
    result
```

---

## 修复后的结果

✓ 编译成功  
✓ 3个高级ML指标正常计算
- LII (流动性失衡指数)
- MTRI (多时间框架共鸣指数)  
- DSRFI (动态支撑阻力破裂指数)

✓ 所有指标值在表格中正确显示 (0-100 范围)

---

## 如何在TradingView中应用修复版本

1. 打开 TradingView → Pine Editor
2. 新建一个 Pine Script (v5)
3. 将修复后的完整代码从 `ML_Order_Helper_V5.pine` 复制
4. 点击 "Add to Chart"
5. 在图表上右键 → "More" → "Toggle Data Window" 查看所有指标值

---

## 相关参考

- [Pine Script v5 函数指南](https://www.tradingview.com/pine-script-docs/en/v5/concepts/Functions.html)
- [Pine Script 用户自定义函数](https://www.tradingview.com/pine-script-docs/en/v5/language/User-defined_functions.html)
- [Pine Script 最佳实践](https://www.tradingview.com/pine-script-docs/en/v5/concepts/Conventions.html)

---

**修复日期**: 2025-12-31  
**版本**: ML_Order_Helper_V5 (修复版)  
**状态**: ✓ 已验证和部署

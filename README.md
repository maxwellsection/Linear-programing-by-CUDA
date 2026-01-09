# 线性规划求解器库 (LP Solver Library)

一个完整的线性规划和混合整数线性规划求解器，基于单纯形法算法实现，支持各种复杂约束和变量类型。

## 🌟 特性

- ✅ **完整的线性规划支持**：单阶段和两阶段单纯形法
- ✅ **混合整数规划 (MILP)**：支持连续变量和二进制变量
- ✅ **复杂约束处理**：≤、≥、= 约束类型
- ✅ **自由变量支持**：可正可负的变量
- ✅ **大M方法**：支持MILP中的大M约束
- ✅ **GPU加速计算**：基于NumPy数组的并行计算
- ✅ **数值稳定性**：容差控制和数值精度处理

## 🚀 快速开始

### 安装

```bash
# 克隆或下载库文件
# 库文件位于 lp_solver/ 目录下
```

### 基本使用

```python
from lp_solver import solve_lp, solve_milp

# 线性规划示例
problem = {
    'objective': {
        'type': 'maximize',
        'coeffs': [3, 2]  # 最大化 3x + 2y
    },
    'constraints': [
        {'type': '<=', 'coeffs': [2, 1], 'rhs': 4},  # 2x + y <= 4
        {'type': '<=', 'coeffs': [1, 2], 'rhs': 4}   # x + 2y <= 4
    ],
    'variables': [
        {'name': 'x', 'type': 'nonneg'},
        {'name': 'y', 'type': 'nonneg'}
    ]
}

result = solve_lp(
    problem['objective'],
    problem['constraints'], 
    problem['variables']
)

print(f"最优解: {result['solution']}")
print(f"最优值: {result['objective_value']}")
```

## 📚 API 参考

### 线性规划求解器

#### `solve_lp(objective, constraints, variables, **kwargs)`

求解线性规划问题。

**参数:**
- `objective` (dict): 目标函数定义
  - `'type'`: 'maximize' 或 'minimize'
  - `'coeffs'`: 系数列表 `[c1, c2, ..., cn]`
- `constraints` (list): 约束条件列表
  - `'type'`: '<=', '>=', 或 '='
  - `'coeffs'`: 系数列表 `[a1, a2, ..., an]`
  - `'rhs'`: 右端值 `b`
  - `'name'` (可选): 约束名称
- `variables` (list): 变量定义列表
  - `'name'`: 变量名
  - `'type'`: 'free', 'nonneg', 'pos', 'neg'
  - `'bounds'` (可选): `[low, high]`

**返回:**
```python
{
    'status': 'optimal' | 'infeasible' | 'unbounded' | 'iterations_exceeded',
    'solution': [x1, x2, ..., xn],  # 最优解
    'objective_value': float,         # 目标函数值
    'solve_time': float,              # 求解时间
    'iterations': int,               # 迭代次数
    'message': str                   # 状态描述
}
```

### 混合整数规划求解器

#### `solve_milp(objective, constraints, variables, **kwargs)`

求解混合整数线性规划问题。

**参数:**
- `objective`, `constraints`: 同线性规划
- `variables` (list): 变量定义列表
  - `'type'`: 'continuous', 'binary', 'free', 'nonneg', 'pos', 'neg'
  - 其他参数同线性规划

**返回:**
```python
{
    'status': 'optimal' | 'infeasible' | 'unbounded' | 'iterations_exceeded',
    'solution': [x1, x2, ..., xn],  # 最优解
    'objective_value': float,         # 目标函数值
    'solve_time': float,              # 求解时间
    'iterations': int,               # 总迭代次数
    'method': str,                   # 求解方法描述
    'message': str                   # 状态描述
}
```

## 🎯 使用示例

### 示例1: 基本线性规划

```python
from lp_solver import solve_lp

# 生产优化问题
problem = {
    'objective': {
        'type': 'maximize',
        'coeffs': [40, 30]  # 产品A利润40，产品B利润30
    },
    'constraints': [
        {'type': '<=', 'coeffs': [2, 1], 'rhs': 100},  # 机器时间限制
        {'type': '<=', 'coeffs': [1, 2], 'rhs': 80},   # 人工时间限制
    ],
    'variables': [
        {'name': 'product_A', 'type': 'nonneg'},
        {'name': 'product_B', 'type': 'nonneg'}
    ]
}

result = solve_lp(problem['objective'], problem['constraints'], problem['variables'])
print(f"最优生产计划: {result['solution']}")
print(f"最大利润: {result['objective_value']}")
```

### 示例2: 混合整数规划 (MILP)

```python
from lp_solver import solve_milp

# 工厂选址问题
problem = {
    'objective': {
        'type': 'minimize',
        'coeffs': [100, 120, 80, 0, 0, 0]  # 运营成本 + 固定成本
    },
    'constraints': [
        # 生产能力约束（包含大M）
        {'type': '<=', 'coeffs': [1, 0, 0, -1000, 0, 0], 'rhs': 0},  # x1 <= 1000*y1
        {'type': '<=', 'coeffs': [0, 1, 0, 0, -1200, 0], 'rhs': 0},   # x2 <= 1200*y2
        {'type': '<=', 'coeffs': [0, 0, 1, 0, 0, -800], 'rhs': 0},    # x3 <= 800*y3
        
        # 市场需求约束
        {'type': '>=', 'coeffs': [1, 1, 1, 0, 0, 0], 'rhs': 500},     # x1 + x2 + x3 >= 500
        
        # 二进制变量约束
        {'type': '<=', 'coeffs': [0, 0, 0, 1, 0, 0], 'rhs': 1},       # y1 <= 1
        {'type': '<=', 'coeffs': [0, 0, 0, 0, 1, 0], 'rhs': 1},       # y2 <= 1
        {'type': '<=', 'coeffs': [0, 0, 0, 0, 0, 1], 'rhs': 1},       # y3 <= 1
    ],
    'variables': [
        {'name': 'factory_1_output', 'type': 'continuous'},
        {'name': 'factory_2_output', 'type': 'continuous'},
        {'name': 'factory_3_output', 'type': 'continuous'},
        {'name': 'factory_1_open', 'type': 'binary'},
        {'name': 'factory_2_open', 'type': 'binary'},
        {'name': 'factory_3_open', 'type': 'binary'},
    ]
}

result = solve_milp(problem['objective'], problem['constraints'], problem['variables'])
print(f"最优解: {result['solution']}")
print(f"最小成本: {result['objective_value']}")
print(f"求解方法: {result['method']}")
```

### 示例3: 自由变量处理

```python
from lp_solver import solve_lp

# 包含自由变量的线性规划
problem = {
    'objective': {
        'type': 'maximize',
        'coeffs': [3, -2, 1]  # 3x1 - 2x2 + x3
    },
    'constraints': [
        {'type': '<=', 'coeffs': [2, 1, -1], 'rhs': 10},
        {'type': '=', 'coeffs': [1, -1, 2], 'rhs': 5},
    ],
    'variables': [
        {'name': 'x1', 'type': 'free'},      # 自由变量（可正可负）
        {'name': 'x2', 'type': 'nonneg'},   # 非负变量
        {'name': 'x3', 'type': 'pos'},      # 正变量
    ]
}

result = solve_lp(problem['objective'], problem['constraints'], problem['variables'])
print(f"最优解: {result['solution']}")
```

## 🔧 高级功能

### 大M方法

在混合整数规划中，大M方法用于处理"如果-那么"约束：

```python
# 示例：生产-库存问题
problem = {
    'objective': {
        'type': 'minimize',
        'coeffs': [5, 3, 100, 80, 60]  # 生产成本 + 设置成本
    },
    'constraints': [
        # 如果生产x1，则必须设置y1=1 (大M约束)
        {'type': '<=', 'coeffs': [1, 0, -100, 0, 0], 'rhs': 0},  # x1 <= 100*y1
        {'type': '<=', 'coeffs': [0, 1, 0, -80, 0], 'rhs': 0},   # x2 <= 80*y2
        
        # 其他业务约束
        {'type': '=', 'coeffs': [1, 1, 0, 0, 1], 'rhs': 150},    # x1 + x2 + x3 = 150
        
        # 二进制变量约束
        {'type': '<=', 'coeffs': [0, 0, 1, 0, 0], 'rhs': 1},     # y1 <= 1
        {'type': '<=', 'coeffs': [0, 0, 0, 1, 0], 'rhs': 1},     # y2 <= 1
    ],
    'variables': [
        {'name': 'product_1', 'type': 'continuous'},
        {'name': 'product_2', 'type': 'continuous'},
        {'name': 'setup_1', 'type': 'binary'},
        {'name': 'setup_2', 'type': 'binary'},
        {'name': 'demand', 'type': 'continuous'},
    ]
}
```

### 变量边界

```python
variables = [
    {'name': 'x1', 'type': 'nonneg', 'bounds': [0, 100]},     # 0 <= x1 <= 100
    {'name': 'x2', 'type': 'free', 'bounds': [-50, 50]},     # -50 <= x2 <= 50
    {'name': 'y1', 'type': 'binary', 'bounds': [0, 1]},      # y1 ∈ {0, 1}
]
```

## ⚡ 性能优化

### 容差设置

```python
# 提高求解精度
result = solve_lp(objective, constraints, variables, tolerance=1e-10)

# 加快求解速度（降低精度要求）
result = solve_lp(objective, constraints, variables, tolerance=1e-6)
```

### 问题预处理

```python
# 1. 标准化约束方向
constraints = [
    {'type': '<=', 'coeffs': [1, 2], 'rhs': 10},  # 保持 <= 约束
    {'type': '>=', 'coeffs': [3, 4], 'rhs': 5},   # 转换为 <=: -3x1 - 4x2 <= -5
]

# 2. 合理设置变量类型
variables = [
    {'name': 'production', 'type': 'nonneg'},  # 生产量通常非负
    {'name': 'setup_decision', 'type': 'binary'},  # 设置决策是二进制的
]
```

## 🐛 常见问题

### Q: 求解器返回"infeasible"怎么办？
A: 检查约束是否相互矛盾，特别是：
- 等式约束是否过约束
- 变量边界是否合理
- 数值计算是否有舍入误差

### Q: 求解时间过长？
A: 尝试：
- 调整容差参数
- 预处理问题（简化约束）
- 使用分支定界法限制搜索深度

### Q: 解不准确？
A: 
- 减小容差参数
- 检查约束条件是否正确设置
- 验证目标函数系数

### Q: 数值不稳定？
A:
- 检查大M值是否过大
- 标准化约束条件
- 使用适当的变量变换

## 📖 算法原理

### 单纯形法
- **单阶段法**：适用于标准形式的线性规划问题
- **两阶段法**：处理等式约束和人工变量

### 分支定界法
1. 求解连续松弛问题
2. 检查整数约束
3. 分支违反约束的变量
4. 递归搜索最优解

### 数值稳定性
- 容差控制
- 枢轴元素检查
- 数值精度处理

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 支持

如有问题，请提交 Issue 或联系维护者。

---

**注意**: 这是一个教学和研究的线性规划求解器实现。在生产环境中使用前，请充分测试其数值稳定性和性能表现。

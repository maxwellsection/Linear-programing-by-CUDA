#!/usr/bin/env python3
"""
高级线性规划求解器
支持复杂约束、自由变量、两阶段单纯形法、无初始可行解处理
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Optional, Union
import warnings

# 尝试导入GPU库
try:
    import cupy as cp
    import torch
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = np
    torch = None

class AdvancedLPSolver:
    """
    高级线性规划求解器
    支持复杂的约束条件、自由变量、两阶段单纯形法
    """
    
    def __init__(self, device='auto', tolerance=1e-10):
        """
        初始化求解器
        
        参数:
            device: 计算设备
            tolerance: 数值容差
        """
        self.tolerance = tolerance
        self.device = device
        
        if device == 'auto':
            if CUDA_AVAILABLE and torch.cuda.is_available():
                self.device = 'cuda'
                print("使用GPU加速")
            else:
                self.device = 'cpu'
                print("使用CPU模式")
        else:
            self.device = device
            print(f"使用{device.upper()}模式")
    
    def solve(self, objective_type, objective_coeffs, constraints, variables_info=None):
        """
        求解复杂的线性规划问题
        
        参数:
            objective_type: 'maximize' 或 'minimize'
            objective_coeffs: 目标函数系数
            constraints: 约束条件列表
            variables_info: 变量信息 [{'name': 'x1', 'type': 'free'/'nonneg'}, ...]
        
        返回:
            求解结果字典
        """
        start_time = time.time()
        
        print(f"\n=== 高级线性规划求解 ===")
        print(f"目标函数: {objective_type} {' + '.join([f'{c:.3f}*x{i+1}' for i, c in enumerate(objective_coeffs)])}")
        print(f"约束数量: {len(constraints)}")
        
        # 1. 约束预处理和标准化
        print("\n1. 约束标准化...")
        standard_form = self._standardize_constraints(constraints)
        
        # 2. 变量处理（自由变量处理）
        print("2. 变量处理...")
        processed_problem = self._process_variables(objective_coeffs, standard_form, variables_info)
        
        # 3. 构建初始单纯形表
        print("3. 构建初始单纯形表...")
        tableau = self._build_initial_tableau(processed_problem)
        
        # 4. 两阶段单纯形法
        print("4. 两阶段单纯形法求解...")
        result = self._two_phase_simplex(tableau, processed_problem, objective_type)
        
        # 5. 后处理和结果提取
        solve_time = time.time() - start_time
        result['solve_time'] = solve_time
        result['device'] = self.device
        
        print(f"求解完成，耗时: {solve_time:.4f}秒")
        
        return result
    
    def _standardize_constraints(self, constraints):
        """
        标准化约束条件
        约束格式: {'type': 'le'/'ge'/'eq', 'coeffs': [a1, a2, ...], 'rhs': b}
        """
        print("标准化约束条件...")
        
        standardized = []
        
        for i, constraint in enumerate(constraints):
            coeffs = np.array(constraint['coeffs'], dtype=float)
            rhs = float(constraint['rhs'])
            ctype = constraint['type'].lower()
            
            # 转换为标准形式 (≤约束)
            if ctype == 'le':
                # 已经是≤形式
                standardized.append({
                    'type': 'le',
                    'coeffs': coeffs,
                    'rhs': rhs,
                    'original_index': i
                })
            elif ctype == 'ge':
                # 乘以-1转换为≤形式
                standardized.append({
                    'type': 'le',
                    'coeffs': -coeffs,
                    'rhs': -rhs,
                    'original_index': i
                })
            elif ctype == 'eq':
                # 等式约束转换为两个不等式
                standardized.append({
                    'type': 'le',
                    'coeffs': coeffs,
                    'rhs': rhs,
                    'original_index': i,
                    'is_equality': True
                })
                standardized.append({
                    'type': 'le',
                    'coeffs': -coeffs,
                    'rhs': -rhs,
                    'original_index': i,
                    'is_equality': True
                })
            else:
                raise ValueError(f"不支持的约束类型: {constraint['type']}")
        
        print(f"标准化完成: {len(constraints)}个原始约束 → {len(standardized)}个标准约束")
        return standardized
    
    def _process_variables(self, objective_coeffs, constraints, variables_info=None):
        """
        处理变量（处理自由变量）
        """
        print("处理变量...")
        
        n_original_vars = len(objective_coeffs)
        var_types = ['nonneg'] * n_original_vars
        
        # 设置变量类型
        if variables_info:
            for i, var_info in enumerate(variables_info):
                if i < n_original_vars:
                    var_types[i] = var_info.get('type', 'nonneg')
        
        # 处理自由变量：将自由变量xi拆分为xi = xi_positive - xi_negative
        processed_coeffs = []
        processed_constraints = []
        variable_mapping = []
        
        for i in range(n_original_vars):
            variable_mapping.append({
                'original_index': i,
                'type': var_types[i],
                'positive_var': len(processed_coeffs),
                'negative_var': None if var_types[i] == 'nonneg' else len(processed_coeffs) + 1
            })
            
            # 添加正变量
            processed_coeffs.append(objective_coeffs[i])
            
            # 如果是自由变量，添加负变量（系数为负）
            if var_types[i] == 'free':
                processed_coeffs.append(-objective_coeffs[i])
        
        # 处理约束
        for constraint in constraints:
            new_coeffs = []
            
            for var_info in variable_mapping:
                orig_idx = var_info['original_index']
                coeff = constraint['coeffs'][orig_idx]
                
                # 添加正变量系数
                new_coeffs.append(coeff)
                
                # 如果是自由变量，还要添加负变量的系数
                if var_info['type'] == 'free':
                    new_coeffs.append(-coeff)
            
            processed_constraints.append({
                'coeffs': np.array(new_coeffs),
                'rhs': constraint['rhs'],
                'original_index': constraint.get('original_index'),
                'is_equality': constraint.get('is_equality', False)
            })
        
        processed_problem = {
            'objective_coeffs': np.array(processed_coeffs),
            'constraints': processed_constraints,
            'variable_mapping': variable_mapping,
            'n_original_vars': n_original_vars,
            'n_processed_vars': len(processed_coeffs)
        }
        
        print(f"变量处理完成: {n_original_vars}个原始变量 → {len(processed_coeffs)}个处理后变量")
        return processed_problem
    
    def _build_initial_tableau(self, problem):
        """
        构建初始单纯形表（包括人工变量）
        """
        print("构建初始单纯形表...")
        
        constraints = problem['constraints']
        obj_coeffs = problem['objective_coeffs']
        n_vars = len(obj_coeffs)
        n_constraints = len(constraints)
        
        # 统计需要的松弛变量和人工变量
        slack_vars_needed = 0
        artificial_vars_needed = 0
        
        for constraint in constraints:
            if constraint['is_equality']:
                # 等式约束需要人工变量
                artificial_vars_needed += 1
            else:
                # 不等式约束需要松弛变量
                slack_vars_needed += 1
        
        # 构建完整表格
        n_total_vars = n_vars + slack_vars_needed + artificial_vars_needed
        tableau = np.zeros((n_constraints + 1, n_total_vars + 1), dtype=float)
        
        var_indices = {
            'original_vars': list(range(n_vars)),
            'slack_vars': list(range(n_vars, n_vars + slack_vars_needed)),
            'artificial_vars': list(range(n_vars + slack_vars_needed, n_total_vars))
        }
        
        # 填充约束系数
        slack_idx = 0
        artificial_idx = 0
        
        for i, constraint in enumerate(constraints):
            # 原始变量系数
            tableau[i, :n_vars] = constraint['coeffs']
            
            # 松弛变量或人工变量
            if constraint['is_equality']:
                # 人工变量（系数为1）
                tableau[i, n_vars + slack_vars_needed + artificial_idx] = 1.0
                artificial_idx += 1
            else:
                # 松弛变量（系数为1）
                tableau[i, n_vars + slack_idx] = 1.0
                slack_idx += 1
            
            # 右端值
            tableau[i, -1] = constraint['rhs']
        
        # 填充目标函数行
        tableau[-1, :n_vars] = -obj_coeffs
        
        problem_info = {
            'tableau': tableau,
            'variable_indices': var_indices,
            'n_constraints': n_constraints,
            'n_vars': n_vars,
            'slack_vars_needed': slack_vars_needed,
            'artificial_vars_needed': artificial_vars_needed
        }
        
        print(f"表格构建完成: {n_constraints}约束, {n_total_vars}变量")
        return problem_info
    
    def _two_phase_simplex(self, problem_info, processed_problem, objective_type):
        """
        两阶段单纯形法
        """
        print("开始两阶段单纯形法...")
        
        tableau = problem_info['tableau'].copy()
        var_indices = problem_info['variable_indices']
        n_constraints = problem_info['n_constraints']
        artificial_vars = var_indices['artificial_vars']
        
        # 阶段1：最小化人工变量的和
        if len(artificial_vars) > 0:
            print("阶段1: 消除人工变量...")
            
            # 构建辅助目标函数（最小化人工变量的和）
            aux_objective = np.zeros(tableau.shape[1])
            aux_objective[artificial_vars] = 1.0
            
            # 更新目标函数行
            for artificial_var in artificial_vars:
                if tableau[artificial_var - var_indices['original_vars'][-1] - 1, artificial_var] != 1.0:
                    # 需要通过行操作消除人工变量
                    artificial_row = artificial_var - var_indices['original_vars'][-1] - 1
                    tableau[-1, :] += tableau[artificial_row, :]
            
            # 迭代优化
            phase1_result = self._simplex_iteration(tableau, artificial_vars, n_constraints, max_iterations=1000)
            
            if phase1_result['status'] != 'optimal':
                return {
                    'status': 'infeasible',
                    'message': '原问题无可行解',
                    'solve_time': 0,
                    'solution': None,
                    'objective_value': None
                }
            
            # 移除人工变量列
            tableau = np.delete(tableau, artificial_vars, axis=1)
            
            # 更新变量索引
            var_indices['artificial_vars'] = []
        
        # 阶段2：原问题求解
        print("阶段2: 求解原问题...")
        
        # 恢复原目标函数
        original_obj = processed_problem['objective_coeffs']
        tableau[-1, :len(original_obj)] = -original_obj
        
        # 如果是最大化问题，转换为最小化
        if objective_type.lower() == 'maximize':
            tableau[-1, :] = -tableau[-1, :]
        
        # 主阶段迭代
        phase2_result = self._simplex_iteration(tableau, [], n_constraints, max_iterations=1000)
        
        # 后处理结果
        solution = self._extract_solution(phase2_result['tableau'], processed_problem, problem_info)
        
        return {
            'status': phase2_result['status'],
            'solution': solution,
            'objective_value': phase2_result.get('objective_value'),
            'tableau': phase2_result['tableau'],
            'iterations': phase2_result.get('iterations', 0),
            'message': phase2_result.get('message', ''),
            'is_feasible': phase2_result['status'] == 'optimal'
        }
    
    def _simplex_iteration(self, tableau, artificial_vars, n_constraints, max_iterations=1000):
        """
        单纯形法主迭代
        """
        iteration = 0
        
        while iteration < max_iterations:
            # 检查最优性
            last_row = tableau[-1, :-1]
            
            # 对于最大化问题，寻找负检验数
            # 对于最小化问题，寻找正检验数
            min_reduced_cost = np.min(last_row)
            max_reduced_cost = np.max(last_row)
            
            if min_reduced_cost >= -self.tolerance:
                # 最优解找到
                objective_value = tableau[-1, -1]
                return {
                    'status': 'optimal',
                    'tableau': tableau,
                    'objective_value': objective_value,
                    'iterations': iteration
                }
            
            if max_reduced_cost <= self.tolerance:
                # 对于最小化问题，这是最优解
                objective_value = tableau[-1, -1]
                return {
                    'status': 'optimal',
                    'tableau': tableau,
                    'objective_value': objective_value,
                    'iterations': iteration
                }
            
            # 选择枢轴列
            pivot_col = np.argmin(last_row)  # 最小化问题选择最小检验数
            
            # 检查无界性
            pivot_col_values = tableau[:-1, pivot_col]
            positive_values = pivot_col_values[pivot_col_values > self.tolerance]
            
            if len(positive_values) == 0:
                return {
                    'status': 'unbounded',
                    'message': '问题无界'
                }
            
            # 选择枢轴行
            ratios = tableau[:-1, -1] / pivot_col_values
            positive_ratios = ratios[pivot_col_values > self.tolerance]
            
            if len(positive_ratios) == 0:
                return {
                    'status': 'unbounded',
                    'message': '问题无界'
                }
            
            # 最小比值规则
            min_ratio_idx = np.argmin(positive_ratios)
            pivot_row = np.where(pivot_col_values > self.tolerance)[0][min_ratio_idx]
            
            # 执行枢轴运算
            tableau = self._pivot_operation(tableau, pivot_row, pivot_col)
            iteration += 1
        
        return {
            'status': 'iterations_exceeded',
            'message': '达到最大迭代次数'
        }
    
    def _pivot_operation(self, tableau, pivot_row, pivot_col):
        """
        执行枢轴运算
        """
        pivot_element = tableau[pivot_row, pivot_col]
        
        # 枢轴行归一化
        tableau[pivot_row, :] /= pivot_element
        
        # 其他行消元
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                multiplier = tableau[i, pivot_col]
                tableau[i, :] -= multiplier * tableau[pivot_row, :]
        
        return tableau
    
    def _extract_solution(self, tableau, processed_problem, problem_info):
        """
        从单纯形表中提取解
        """
        solution = np.zeros(processed_problem['n_original_vars'])
        variable_mapping = processed_problem['variable_mapping']
        
        for var_info in variable_mapping:
            orig_idx = var_info['original_index']
            
            # 检查基变量
            for i in range(tableau.shape[0] - 1):
                col = tableau[i, :-1]
                if np.sum(np.abs(col)) == 1.0:
                    basis_var_idx = np.argmax(col == 1.0)
                    if basis_var_idx == var_info['positive_var']:
                        solution[orig_idx] = tableau[i, -1]
                    elif var_info['type'] == 'free' and basis_var_idx == var_info['negative_var']:
                        solution[orig_idx] -= tableau[i, -1]
        
        return solution


# 便捷函数
def solve_complex_lp(objective_type, objective_coeffs, constraints, variables_info=None):
    """
    便捷函数：求解复杂线性规划问题
    """
    solver = AdvancedLPSolver()
    return solver.solve(objective_type, objective_coeffs, constraints, variables_info)


# 示例和测试
if __name__ == "__main__":
    print("高级线性规划求解器测试")
    print("=" * 60)
    
    # 测试复杂示例：生产计划问题
    print("\n=== 复杂生产计划问题 ===")
    
    # 目标函数：最大化利润
    objective_type = 'maximize'
    objective_coeffs = [40.0, 30.0, 20.0]  # 3种产品的利润
    
    # 复杂约束条件
    constraints = [
        # 机器时间限制
        {'type': 'le', 'coeffs': [2, 1, 1], 'rhs': 100},
        {'type': 'ge', 'coeffs': [1, 2, 1], 'rhs': 60},  # 最小生产要求
        {'type': 'eq', 'coeffs': [1, 1, 2], 'rhs': 80},   # 人工时间精确限制
        # 原材料限制
        {'type': 'le', 'coeffs': [3, 2, 1], 'rhs': 150},
        # 质量约束
        {'type': 'ge', 'coeffs': [1, 0, 1], 'rhs': 20},   # 至少生产产品1和3
        # 混合约束
        {'type': 'le', 'coeffs': [0, 1, 2], 'rhs': 40}
    ]
    
    # 变量信息：x1, x2自由变量，x3非负
    variables_info = [
        {'name': 'x1', 'type': 'free'},      # 可以为负
        {'name': 'x2', 'type': 'nonneg'},   # 非负
        {'name': 'x3', 'type': 'nonneg'}    # 非负
    ]
    
    # 求解
    result = solve_complex_lp(objective_type, objective_coeffs, constraints, variables_info)
    
    print(f"\n求解结果:")
    print(f"状态: {result['status']}")
    print(f"是否可行: {result['is_feasible']}")
    print(f"求解时间: {result['solve_time']:.4f}秒")
    
    if result['status'] == 'optimal':
        solution = result['solution']
        print(f"最优解:")
        for i, val in enumerate(solution):
            print(f"  x{i+1} = {val:.4f}")
        print(f"最优值: {result['objective_value']:.4f}")
        
        # 验证约束满足性
        print(f"\n约束验证:")
        for i, constraint in enumerate(constraints):
            lhs = sum(solution[j] * constraint['coeffs'][j] for j in range(len(constraint['coeffs'])))
            rhs = constraint['rhs']
            
            if constraint['type'] == 'le':
                satisfied = lhs <= rhs + 1e-6
                print(f"  约束{i+1}: {lhs:.4f} ≤ {rhs:.4f} {'✓' if satisfied else '✗'}")
            elif constraint['type'] == 'ge':
                satisfied = lhs >= rhs - 1e-6
                print(f"  约束{i+1}: {lhs:.4f} ≥ {rhs:.4f} {'✓' if satisfied else '✗'}")
            elif constraint['type'] == 'eq':
                satisfied = abs(lhs - rhs) <= 1e-6
                print(f"  约束{i+1}: {lhs:.4f} = {rhs:.4f} {'✓' if satisfied else '✗'}")
    
    elif result['status'] == 'infeasible':
        print("❌ 问题无可行解")
    elif result['status'] == 'unbounded':
        print("❌ 问题无界")
    else:
        print(f"❌ 求解失败: {result.get('message', '未知错误')}")

import matplotlib
matplotlib.use('TkAgg')
from pyvrp import Model
from pyvrp.stop import MaxRuntime, MaxIterations, NoImprovement, MultipleCriteria, StoppingCriterion
from pyvrp.search import (
    NODE_OPERATORS,
    ROUTE_OPERATORS,
    LocalSearch,
    NeighbourhoodParams,
    compute_neighbours,
)
from pyvrp.minimise_fleet import _lower_bound
import math
import re
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
import time

# 导入自定义的混合策略算法
from mab_genetic import create_hybrid_genetic_algorithm, create_island_genetic_algorithm

# 混合策略参数配置
HYBRID_CONFIG = {
    'switch_interval': 2000,  # 每100次迭代评估一次性能并决定是否切换策略
    'tournament_size': 5,    # 锦标赛选择器的大小
    'num_arms': 8,           # MAB选择器的臂数量
    'c': 1.2,                # MAB的探索参数
    'initial_temperature': 1.5,  # 初始温度
    'cooling_rate': 0.9998,  # 冷却速率
    'custom_vehicle_weight': 5.0,  # 车辆数权重
    'custom_distance_weight': 0.0   # 距离权重
}

# 岛屿模型参数配置
ISLAND_CONFIG = {
    'num_islands': 16, 
    'migration_interval': 500, # 迁移间隔（迭代次数）- 增加间隔减少迁移频率
    'migration_queue_size': 50,  # 迁移队列大小 - 减少队列大小
    'diversity_migration_threshold': 200,  # 开始多样性迁移的停滞阈值
    'min_isolation_period': 1000  # 最小隔离期 - 新增参数
}

# 定义全局缩放因子，用于保留三位小数精度
SCALE_FACTOR = 10000

class EarlyStopAfterFeasible:
    """
    自定义停止条件：在找到可行解后延迟指定时间停止
    支持两种调用方式：传统的成本检查和结果对象检查
    """
    def __init__(self, max_runtime, delay_after_feasible=10):
        """
        参数:
        - max_runtime: 最大运行时间（秒）
        - delay_after_feasible: 找到可行解后的延迟时间（秒），默认10秒
        """
        self.max_runtime = max_runtime
        self.delay_after_feasible = delay_after_feasible
        self.start_time = None
        self.feasible_found_time = None
        self.population_reference = None  # 用于获取最佳解
        
    def set_population_reference(self, population, cost_evaluator):
        """设置种群引用，用于检查可行性"""
        self.population_reference = population
        self.cost_evaluator_reference = cost_evaluator
        
    def __call__(self, cost_or_result):
        """
        检查是否应该停止
        支持两种调用方式：
        1. cost_or_result 是一个数字（成本）- 传统调用方式
        2. cost_or_result 是一个结果对象 - 新的调用方式
        """
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # 检查是否超过最大运行时间
        if elapsed_time >= self.max_runtime:
            return True
        
        # 判断调用方式并检查是否找到可行解
        has_feasible_solution = False
        
        try:
            # 尝试作为结果对象处理
            if hasattr(cost_or_result, 'best'):
                has_feasible_solution = cost_or_result.best.is_feasible()
            # 尝试作为成本值处理，需要通过其他方式检查可行性
            elif self.population_reference is not None:
                # 通过种群获取最佳解
                best_solution = self.population_reference.best()
                if best_solution is not None:
                    has_feasible_solution = best_solution.is_feasible()
        except:
            # 如果无法确定可行性，继续运行
            has_feasible_solution = False
            
        # 如果找到可行解，处理延迟停止逻辑
        if has_feasible_solution:
            if self.feasible_found_time is None:
                # 第一次找到可行解，记录时间
                self.feasible_found_time = current_time
                print(f"找到可行解！将在 {self.delay_after_feasible} 秒后停止...")
                
            # 检查是否已经延迟了足够的时间
            elif current_time - self.feasible_found_time >= self.delay_after_feasible:
                print(f"延迟 {self.delay_after_feasible} 秒后停止")
                return True
                
        return False

def load_solomon_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    vehicle_info_str = re.search(r'VEHICLE\nNUMBER\s+CAPACITY\n\s*(\d+)\s+(\d+)', content)
    num_vehicles = int(vehicle_info_str.group(1))
    vehicle_capacity = int(vehicle_info_str.group(2))
    customer_data_str = re.search(r'CUSTOMER\n(.+)', content, re.DOTALL).group(1)
    lines = customer_data_str.strip().split('\n')[2:]
    coords, demands, time_windows, service_times = [], [], [], []
    for line in lines:
        if not line.strip(): continue
        parts = list(map(int, line.split()))
        coords.append((parts[1], parts[2]))
        demands.append(parts[3])
        time_windows.append((parts[4], parts[5]))
        service_times.append(parts[6])
    return num_vehicles, vehicle_capacity, coords, demands, time_windows, service_times


def validate_pyvrp_solution(solution, num_all_clients: int, vehicle_capacity: int, filename: str, scale_factor: int = SCALE_FACTOR):
    """
    验证PyVRP解决方案的可行性，同时支持浮点精度的距离和时间计算
    
    参数:
    - solution: PyVRP解决方案对象
    - num_all_clients: 客户总数
    - vehicle_capacity: 车辆容量限制
    - filename: 数据文件名（用于日志输出）
    - scale_factor: 距离和时间的缩放因子，默认为全局缩放因子
    
    返回:
    - bool: 解决方案是否有效
    """
    print(f"正在为文件 {filename} 验证方案 (Validating solution for file {filename})...")
    all_required_clients = set(range(1, num_all_clients + 1))
    served_clients = set()
    for route_idx, route in enumerate(solution.routes()):
        for client in route.visits():
            if client in served_clients:
                print(f"错误 (Error): 客户 (Customer) {client} 被服务了多次 (served multiple times)。")
                return False
            served_clients.add(client)
    unserved = all_required_clients - served_clients
    if unserved:
        print(f"警告 (Warning): 客户 (Customers) {unserved} 未被服务 (not served)。")
    for route_idx, route in enumerate(solution.routes()):
        vehicle_num = route_idx + 1
        total_demand = sum(route.delivery())
        if total_demand > vehicle_capacity:
            print(f"错误 (Error): 车辆 (Vehicle) {vehicle_num} 的路径超载 (exceeds capacity)。")
            print(f" -> 路径需求 (Route Demand): {total_demand}, 车辆容量 (Vehicle Capacity): {vehicle_capacity}")
            return False
        
        # 时间窗冲突检查时考虑缩放因子
        time_warp = route.time_warp()
        if time_warp > 0:
            # 将时间窗冲突值转换回实际值（考虑浮点精度）
            actual_time_warp = time_warp / scale_factor
            print(f"错误 (Error): 车辆 (Vehicle) {vehicle_num} 的路径违反了时间窗 (violates time windows)。")
            print(f" -> 时间窗冲突值 (Time Warp Value): {actual_time_warp:.2f}")
            return False
            
    print(f"文件 {filename} 的方案验证成功 (Solution validation for file {filename} completed successfully!)")
    return True

def get_island_specific_config(island_id, base_config):
    """为不同岛屿动态生成多样化的配置参数，确保岛屿间的高度差异性
    
    使用岛屿ID的数学特性来生成不同的参数组合，确保：
    1. 每个岛屿都有独特且显著不同的参数组合  
    2. 参数在合理范围内变化，涵盖不同的搜索策略
    3. 支持任意数量的岛屿，避免参数重复
    4. 包含邻域搜索和局部搜索的多样化配置
    """
    import math
    import hashlib
    
    # 使用岛屿ID生成伪随机种子，确保可重复性
    seed = int(hashlib.md5(f"island_{island_id}".encode()).hexdigest()[:8], 16)
    
    # 设置随机种子以确保每个岛屿的配置是可重复的
    import random
    temp_random = random.Random(seed)
    
    # 大幅扩展参数范围，确保更大的多样性
    param_ranges = {
        'tournament_size': (2, 12),      # 大幅扩展锦标赛大小范围
        'num_arms': (3, 25),             # 大幅扩展MAB臂数范围  
        'c': (0.3, 4.0),                 # 大幅扩展探索参数范围
        'switch_interval': (30, 300),    # 大幅扩展切换间隔范围
        'initial_temperature': (0.5, 5.0),  # 大幅扩展初始温度范围
        'cooling_rate': (0.9990, 0.99995),  # 扩展冷却速率范围
        'custom_vehicle_weight': (1.0, 15.0),  # 车辆权重范围
        'custom_distance_weight': (0.0, 5.0)   # 距离权重范围
    }
    
    # 生成基础配置
    island_config = base_config.copy()
    
    # 为每个参数生成值
    for param, (min_val, max_val) in param_ranges.items():
        if param in ['tournament_size', 'num_arms', 'switch_interval']:
            # 整数参数
            island_config[param] = temp_random.randint(int(min_val), int(max_val))
        else:
            # 浮点参数
            island_config[param] = temp_random.uniform(min_val, max_val)
    
    # 定义10种不同的策略类型，确保更大差异性
    strategy_type = island_id % 10
    
    if strategy_type == 0:  # 极保守策略
        island_config.update({
            'tournament_size': max(2, island_config['tournament_size'] // 2),
            'c': min(0.8, island_config['c'] * 0.5),
            'switch_interval': max(200, island_config['switch_interval']),
            'initial_temperature': min(1.0, island_config['initial_temperature']),
            'custom_vehicle_weight': min(3.0, island_config['custom_vehicle_weight']),
            'num_arms': min(6, island_config['num_arms'])
        })
        
    elif strategy_type == 1:  # 极激进策略
        island_config.update({
            'tournament_size': min(12, island_config['tournament_size'] + 3),
            'c': max(2.5, island_config['c'] * 1.5),
            'switch_interval': min(50, island_config['switch_interval']),
            'initial_temperature': max(3.0, island_config['initial_temperature']),
            'custom_vehicle_weight': max(8.0, island_config['custom_vehicle_weight']),
            'num_arms': max(15, island_config['num_arms'])
        })
        
    elif strategy_type == 2:  # 高多样性MAB策略
        island_config.update({
            'num_arms': max(20, island_config['num_arms']),
            'c': max(2.0, island_config['c']),
            'switch_interval': temp_random.randint(60, 120),
            'initial_temperature': temp_random.uniform(2.0, 4.0),
            'custom_distance_weight': max(2.0, island_config['custom_distance_weight'])
        })
        
    elif strategy_type == 3:  # 锦标赛主导策略
        island_config.update({
            'tournament_size': max(8, island_config['tournament_size']),
            'switch_interval': max(150, island_config['switch_interval']),
            'c': min(1.2, island_config['c']),
            'custom_vehicle_weight': temp_random.uniform(10.0, 15.0),
            'num_arms': min(8, island_config['num_arms'])
        })
        
    elif strategy_type == 4:  # 距离优化专家
        island_config.update({
            'custom_distance_weight': max(3.0, island_config['custom_distance_weight']),
            'custom_vehicle_weight': min(2.0, island_config['custom_vehicle_weight']),
            'c': temp_random.uniform(1.0, 2.0),
            'switch_interval': temp_random.randint(80, 150),
            'initial_temperature': temp_random.uniform(1.5, 2.5)
        })
        
    elif strategy_type == 5:  # 车辆优化专家
        island_config.update({
            'custom_vehicle_weight': max(10.0, island_config['custom_vehicle_weight']),
            'custom_distance_weight': 0.0,
            'c': max(1.8, island_config['c']),
            'initial_temperature': max(2.5, island_config['initial_temperature']),
            'switch_interval': min(80, island_config['switch_interval'])
        })
        
    elif strategy_type == 6:  # 高温长时策略
        island_config.update({
            'initial_temperature': max(4.0, island_config['initial_temperature']),
            'cooling_rate': min(0.99992, island_config['cooling_rate']),
            'switch_interval': max(200, island_config['switch_interval']),
            'c': temp_random.uniform(1.5, 3.0)
        })
        
    elif strategy_type == 7:  # 快速切换策略
        island_config.update({
            'switch_interval': min(40, island_config['switch_interval']),
            'c': max(2.0, island_config['c']),
            'tournament_size': temp_random.randint(6, 10),
            'initial_temperature': temp_random.uniform(1.8, 3.0)
        })
        
    elif strategy_type == 8:  # 均衡探索策略
        island_config.update({
            'custom_vehicle_weight': temp_random.uniform(4.0, 8.0),
            'custom_distance_weight': temp_random.uniform(1.0, 3.0),
            'c': temp_random.uniform(1.2, 2.2),
            'switch_interval': temp_random.randint(90, 150)
        })
        
    else:  # strategy_type == 9: 动态自适应策略
        island_config.update({
            'switch_interval': temp_random.randint(40, 80),  # 频繁切换
            'c': temp_random.uniform(1.5, 3.5),
            'initial_temperature': temp_random.uniform(2.0, 4.5),
            'cooling_rate': temp_random.uniform(0.9991, 0.9998),
            'num_arms': temp_random.randint(12, 25)
        })
    
    # 额外的细粒度随机化，基于岛屿ID的不同位
    fine_tune_seed = (island_id * 7 + 13) % 20
    
    # 微调因子数组
    temp_multipliers = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 
                       1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    weight_multipliers = [0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2,
                         2.5, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0]
    
    # 应用微调
    island_config['initial_temperature'] *= temp_multipliers[fine_tune_seed]
    island_config['custom_vehicle_weight'] *= weight_multipliers[fine_tune_seed]
    
    # 进一步的距离权重调整
    distance_additions = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5, 3.5]
    distance_idx = (island_id * 3) % len(distance_additions)
    island_config['custom_distance_weight'] += distance_additions[distance_idx]
    
    # 最终边界检查，确保参数在合理范围内
    island_config['tournament_size'] = max(2, min(15, int(island_config['tournament_size'])))
    island_config['num_arms'] = max(3, min(30, int(island_config['num_arms'])))
    island_config['c'] = max(0.1, min(5.0, island_config['c']))
    island_config['switch_interval'] = max(20, min(400, int(island_config['switch_interval'])))
    island_config['initial_temperature'] = max(0.1, min(8.0, island_config['initial_temperature']))
    island_config['cooling_rate'] = max(0.9985, min(0.99999, island_config['cooling_rate']))
    island_config['custom_vehicle_weight'] = max(0.5, min(20.0, island_config['custom_vehicle_weight']))
    island_config['custom_distance_weight'] = max(0.0, min(8.0, island_config['custom_distance_weight']))
    
    return island_config

def run_island_worker_with_migration(island_id, filepath, num_vehicles, runtime_limit, seed_base, hybrid_params, migration_queue, result_queue):
    """
    支持迁移机制的岛屿模型工作进程函数
    
    参数:
    - island_id: 岛屿编号
    - filepath: 数据文件路径
    - num_vehicles: 车辆数量
    - runtime_limit: 运行时间限制
    - seed_base: 基础随机种子
    - hybrid_params: 混合算法参数
    - migration_queue: 迁移队列
    - result_queue: 结果队列
    
    返回:
    - (island_id, best_cost, solution_data) 或 None
    """
    try:
        # print(f"[岛屿 {island_id}] 开始处理（支持迁移）...")
        
        # 为每个岛屿使用不同的随机种子
        island_seed = seed_base + island_id * 1000
        
        # 为每个岛屿生成专属的混合策略参数
        island_hybrid_params = get_island_specific_config(island_id, hybrid_params)
        
        
        # 使用全局缩放因子
        scale_factor = SCALE_FACTOR
        
        # 重新加载数据（每个进程独立加载）
        INITIAL_NUM_VEHICLES, VEHICLE_CAPACITY, COORDS, DEMANDS, TIME_WINDOWS, SERVICE_TIMES = load_solomon_data(filepath)
        num_locations = len(COORDS)
        
        # 缩放时间相关参数
        SCALED_TIME_WINDOWS = []
        SCALED_SERVICE_TIMES = []
        
        for i in range(len(TIME_WINDOWS)):
            early, late = TIME_WINDOWS[i]
            SCALED_TIME_WINDOWS.append((int(early * scale_factor), int(late * scale_factor)))
            SCALED_SERVICE_TIMES.append(int(SERVICE_TIMES[i] * scale_factor))
        
        # 计算持续时间并缩放
        DURATION_MATRIX = [[math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) * scale_factor for c2 in COORDS] for c1 in COORDS]
        
        # 构建模型
        m = Model()
        m.add_vehicle_type(num_vehicles, capacity=VEHICLE_CAPACITY, tw_early=SCALED_TIME_WINDOWS[0][0], tw_late=SCALED_TIME_WINDOWS[0][1])
        m.add_depot(x=COORDS[0][0], y=COORDS[0][1], tw_early=SCALED_TIME_WINDOWS[0][0], tw_late=SCALED_TIME_WINDOWS[0][1])
        
        # 添加客户
        clients = [m.add_client(
            x=COORDS[i][0], 
            y=COORDS[i][1], 
            tw_early=SCALED_TIME_WINDOWS[i][0], 
            tw_late=SCALED_TIME_WINDOWS[i][1], 
            delivery=DEMANDS[i], 
            service_duration=SCALED_SERVICE_TIMES[i]
        ) for i in range(1, len(COORDS))]
        
        # 添加边
        for i in range(num_locations):
            for j in range(num_locations):
                frm = m.locations[i]
                to = m.locations[j]
                distance = math.sqrt((frm.x - to.x)**2 + (frm.y - to.y)**2)
                scaled_distance = int(distance * scale_factor)
                scaled_duration = int(DURATION_MATRIX[i][j])
                m.add_edge(frm, to, distance=scaled_distance, duration=scaled_duration)
        
        # 创建支持迁移的岛屿遗传算法
        data = m.data()
        algo, rng = create_island_genetic_algorithm(
            data=data,
            seed=island_seed,
            hybrid_params=island_hybrid_params,  # 使用岛屿专属配置
            custom_neighbourhood_params=NeighbourhoodParams(),  # 默认参数，实际由多样化配置生成器处理
            custom_swap_star_params=0.1,  # 保留用于后备，实际由多样化配置生成器处理
            island_id=island_id,
            migration_queue=migration_queue,
            is_distance_optimization=False  # 车辆优化阶段
        )
        
        # 设置迁移间隔和最小隔离期
        algo.migration_interval = ISLAND_CONFIG['migration_interval']
        algo.min_isolation_period = ISLAND_CONFIG['min_isolation_period']
        
        # 运行算法 - 使用早期停止条件
        stop_criteria = EarlyStopAfterFeasible(
            max_runtime=runtime_limit, 
            delay_after_feasible=10
        )
        res = algo.run(stop=stop_criteria, collect_stats=False, display=False)
        
        if res.best.is_feasible():
            best_cost = res.best.distance() / scale_factor
            
            # 序列化解决方案数据
            routes_data = []
            for route in res.best.routes():
                route_visits = list(route.visits())
                routes_data.append(route_visits)
            
            solution_data = {
                'routes': routes_data,
                'distance': best_cost,
                'is_feasible': True,
                'num_vehicles': len(res.best.routes())
            }
            
            print(f"[岛屿 {island_id}] 完成，找到可行解，成本: {best_cost:.2f}")
            result_queue.put((island_id, best_cost, solution_data))
            return (island_id, best_cost, solution_data)
        else:
            print(f"[岛屿 {island_id}] 完成，未找到可行解")
            return None
            
    except Exception as e:
        print(f"[岛屿 {island_id}] 错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_island_worker(island_id, filepath, num_vehicles, runtime_limit, seed_base, hybrid_params, island_config):
    """
    岛屿模型工作进程函数（保持向后兼容）
    
    参数:
    - island_id: 岛屿编号
    - filepath: 数据文件路径
    - num_vehicles: 车辆数量
    - runtime_limit: 运行时间限制
    - seed_base: 基础随机种子
    - hybrid_params: 混合算法参数
    - island_config: 岛屿配置参数
    
    返回:
    - (island_id, best_cost, solution_data) 或 None
    """
    try:
        print(f"[岛屿 {island_id}] 开始处理...")
        
        # 为每个岛屿使用不同的随机种子
        island_seed = seed_base + island_id * 1000
        
        # 使用全局缩放因子
        scale_factor = SCALE_FACTOR
        
        # 重新加载数据（每个进程独立加载）
        INITIAL_NUM_VEHICLES, VEHICLE_CAPACITY, COORDS, DEMANDS, TIME_WINDOWS, SERVICE_TIMES = load_solomon_data(filepath)
        num_locations = len(COORDS)
        
        # 缩放时间相关参数
        SCALED_TIME_WINDOWS = []
        SCALED_SERVICE_TIMES = []
        
        for i in range(len(TIME_WINDOWS)):
            early, late = TIME_WINDOWS[i]
            SCALED_TIME_WINDOWS.append((int(early * scale_factor), int(late * scale_factor)))
            SCALED_SERVICE_TIMES.append(int(SERVICE_TIMES[i] * scale_factor))
        
        # 计算持续时间并缩放
        DURATION_MATRIX = [[math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) * scale_factor for c2 in COORDS] for c1 in COORDS]
        
        # 构建模型
        m = Model()
        m.add_vehicle_type(num_vehicles, capacity=VEHICLE_CAPACITY, tw_early=SCALED_TIME_WINDOWS[0][0], tw_late=SCALED_TIME_WINDOWS[0][1])
        m.add_depot(x=COORDS[0][0], y=COORDS[0][1], tw_early=SCALED_TIME_WINDOWS[0][0], tw_late=SCALED_TIME_WINDOWS[0][1])
        
        # 添加客户
        clients = [m.add_client(
            x=COORDS[i][0], 
            y=COORDS[i][1], 
            tw_early=SCALED_TIME_WINDOWS[i][0], 
            tw_late=SCALED_TIME_WINDOWS[i][1], 
            delivery=DEMANDS[i], 
            service_duration=SCALED_SERVICE_TIMES[i]
        ) for i in range(1, len(COORDS))]
        
        # 添加边
        for i in range(num_locations):
            for j in range(num_locations):
                frm = m.locations[i]
                to = m.locations[j]
                distance = math.sqrt((frm.x - to.x)**2 + (frm.y - to.y)**2)
                scaled_distance = int(distance * scale_factor)
                scaled_duration = int(DURATION_MATRIX[i][j])
                m.add_edge(frm, to, distance=scaled_distance, duration=scaled_duration)
        
        # 为每个岛屿生成专属的混合策略参数
        island_hybrid_params = get_island_specific_config(island_id, hybrid_params)
        
        # 创建遗传算法
        data = m.data()
        algo, rng = create_hybrid_genetic_algorithm(
            data=data,
            seed=island_seed,
            hybrid_params=island_hybrid_params,  # 使用岛屿专属配置
            custom_neighbourhood_params=NeighbourhoodParams(),  # 默认参数，实际由多样化配置生成器处理
            custom_swap_star_params=0.1,  # 保留用于后备，实际由多样化配置生成器处理
            island_id=island_id  # 传递岛屿ID以启用多样化初始解和搜索配置
        )
        
        # 运行算法 - 使用早期停止条件
        stop_criteria = EarlyStopAfterFeasible(
            max_runtime=runtime_limit, 
            delay_after_feasible=10
        )
        res = algo.run(stop=stop_criteria, collect_stats=False, display=False)
        
        if res.best.is_feasible():
            best_cost = res.best.distance() / scale_factor
            
            # 序列化解决方案数据
            routes_data = []
            for route in res.best.routes():
                route_visits = list(route.visits())
                routes_data.append(route_visits)
            
            solution_data = {
                'routes': routes_data,
                'distance': best_cost,
                'is_feasible': True,
                'num_vehicles': len(res.best.routes())
            }
            
            print(f"[岛屿 {island_id}] 完成，找到可行解，成本: {best_cost:.2f}")
            return (island_id, best_cost, solution_data)
        else:
            print(f"[岛屿 {island_id}] 完成，未找到可行解")
            return None
            
    except Exception as e:
        print(f"[岛屿 {island_id}] 错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def solve_with_island_model(filepath, num_vehicles, runtime_limit, seed_base=42):
    """
    使用岛屿模型求解VRP问题（支持真正的迁移机制）
    
    参数:
    - filepath: 数据文件路径
    - num_vehicles: 车辆数量
    - runtime_limit: 运行时间限制
    - seed_base: 基础随机种子
    
    返回:
    - 最优解数据或None
    """
    import multiprocessing as mp
    from multiprocessing import Process, Queue
    
    num_islands = ISLAND_CONFIG['num_islands']
    # print(f"启动岛屿模型求解（支持迁移），{num_islands}个岛屿并行...")
    
    # 创建共享的迁移队列
    migration_queue = Queue(maxsize=ISLAND_CONFIG['migration_queue_size'])  # 使用配置的队列大小
    result_queue = Queue()
    
    # 启动岛屿进程
    processes = []
    try:
        for island_id in range(num_islands):
            p = Process(
                target=run_island_worker_with_migration,
                args=(
                    island_id, 
                    filepath, 
                    num_vehicles, 
                    runtime_limit, 
                    seed_base, 
                    HYBRID_CONFIG,
                    migration_queue,  # 传入共享的迁移队列
                    result_queue     # 结果队列
                )
            )
            p.start()
            processes.append(p)
            # print(f"启动岛屿 {island_id}（支持迁移）")
        
                # 等待所有进程完成
        for p in processes:
            p.join()
        
        # 收集结果
        results = []
        while not result_queue.empty():
            try:
                result = result_queue.get(timeout=1.0)
                results.append(result)
            except:
                break
        
        # 过滤掉None结果
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            # 找到最优解
            best_result = min(valid_results, key=lambda x: x[1])
            island_id, best_cost, solution_data = best_result
            
            print(f"岛屿模型求解完成！最优解来自岛屿 {island_id}，成本: {best_cost:.2f}")
            print(f"迁移队列剩余项目: {migration_queue.qsize()}")
            return solution_data
        else:
            print("岛屿模型求解完成，但所有岛屿都未找到可行解")
            return None
            
    except Exception as e:
        print(f"岛屿模型求解时出错: {e}")
        return None
    finally:
        # 清理进程
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()


def run_distance_optimization_with_islands(filepath, min_vehicles, runtime_limit, seed_base=42, elite_solutions=None):
    """
    使用岛屿模型进行距离优化阶段
    
    参数:
    - filepath: 数据文件路径
    - min_vehicles: 固定的最小车辆数
    - runtime_limit: 运行时间限制
    - seed_base: 基础随机种子
    - elite_solutions: 来自车辆优化阶段的精英解
    
    返回:
    - 最优解数据或None
    """
    import multiprocessing as mp
    from multiprocessing import Process, Queue
    
    num_islands = ISLAND_CONFIG['num_islands']
    # print(f"启动距离优化岛屿模型，{num_islands}个岛屿并行...")
    
    # 创建共享的迁移队列
    migration_queue = Queue(maxsize=max(20, ISLAND_CONFIG['migration_queue_size'] // 3))  # 距离优化阶段队列更小
    result_queue = Queue()
    
    # 如果有精英解，预先填充到迁移队列中
    if elite_solutions:
        print(f"注入{len(elite_solutions)}个精英解到距离优化阶段")
        for elite in elite_solutions:
            try:
                migration_queue.put(elite, timeout=0.1)
            except:
                break
    
    # 启动岛屿进程
    processes = []
    try:
        for island_id in range(num_islands):
            p = Process(
                target=run_distance_optimization_island,
                args=(
                    island_id, 
                    filepath, 
                    min_vehicles, 
                    runtime_limit, 
                    seed_base + 100,  # 使用不同的种子基数
                    migration_queue,
                    result_queue
                )
            )
            p.start()
            processes.append(p)
            # print(f"启动距离优化岛屿 {island_id}")
        
                # 等待所有进程完成
        for p in processes:
            p.join()

        
        # 收集结果
        results = []
        while not result_queue.empty():
            try:
                result = result_queue.get(timeout=1.0)
                results.append(result)
            except:
                break
        
        # 过滤掉None结果
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            # 找到最优解
            best_result = min(valid_results, key=lambda x: x[1])
            island_id, best_cost, solution_data = best_result
            
            print(f"距离优化岛屿模型完成！最优解来自岛屿 {island_id}，成本: {best_cost:.2f}")
            return solution_data
        else:
            print("距离优化岛屿模型完成，但所有岛屿都未找到可行解")
            return None
            
    except Exception as e:
        print(f"距离优化岛屿模型求解时出错: {e}")
        return None
    finally:
        # 清理进程
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    p.kill()


def run_distance_optimization_island(island_id, filepath, min_vehicles, runtime_limit, seed_base, migration_queue, result_queue):
    """
    距离优化阶段的岛屿工作进程
    """
    try:
        # print(f"[距离优化岛屿 {island_id}] 开始处理...")
        
        # 为每个岛屿使用不同的随机种子
        island_seed = seed_base + island_id * 1000
        
        # 距离优化阶段的基础参数
        base_distance_params = HYBRID_CONFIG.copy()
        base_distance_params['custom_vehicle_weight'] = 0
        base_distance_params['custom_distance_weight'] = 5.0
        
        # 为距离优化阶段生成专属配置
        island_distance_params = get_island_specific_config(island_id, base_distance_params)
        # 距离优化阶段的特殊调整
        island_distance_params['num_arms'] = min(12, island_distance_params['num_arms'] + 2)  # 增加臂数
        island_distance_params['c'] = max(0.5, island_distance_params['c'] - 0.2)  # 降低探索参数
        island_distance_params['switch_interval'] = max(50, island_distance_params['switch_interval'] - 20)  # 更频繁切换
        
        # print(f"[距离优化岛屿 {island_id}] 配置: T{island_distance_params['tournament_size']}_A{island_distance_params['num_arms']}_C{island_distance_params['c']:.1f}_I{island_distance_params['switch_interval']}")
        
        # 使用全局缩放因子
        scale_factor = SCALE_FACTOR
        
        # 重新加载数据
        INITIAL_NUM_VEHICLES, VEHICLE_CAPACITY, COORDS, DEMANDS, TIME_WINDOWS, SERVICE_TIMES = load_solomon_data(filepath)
        num_locations = len(COORDS)
        
        # 缩放时间相关参数
        SCALED_TIME_WINDOWS = []
        SCALED_SERVICE_TIMES = []
        
        for i in range(len(TIME_WINDOWS)):
            early, late = TIME_WINDOWS[i]
            SCALED_TIME_WINDOWS.append((int(early * scale_factor), int(late * scale_factor)))
            SCALED_SERVICE_TIMES.append(int(SERVICE_TIMES[i] * scale_factor))
        
        # 计算持续时间并缩放
        DURATION_MATRIX = [[math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) * scale_factor for c2 in COORDS] for c1 in COORDS]
        
        # 构建模型（固定车辆数）
        m = Model()
        m.add_vehicle_type(min_vehicles, capacity=VEHICLE_CAPACITY, tw_early=SCALED_TIME_WINDOWS[0][0], tw_late=SCALED_TIME_WINDOWS[0][1])
        m.add_depot(x=COORDS[0][0], y=COORDS[0][1], tw_early=SCALED_TIME_WINDOWS[0][0], tw_late=SCALED_TIME_WINDOWS[0][1])
        
        # 添加客户
        clients = [m.add_client(
            x=COORDS[i][0], 
            y=COORDS[i][1], 
            tw_early=SCALED_TIME_WINDOWS[i][0], 
            tw_late=SCALED_TIME_WINDOWS[i][1], 
            delivery=DEMANDS[i], 
            service_duration=SCALED_SERVICE_TIMES[i]
        ) for i in range(1, len(COORDS))]
        
        # 添加边
        for i in range(num_locations):
            for j in range(num_locations):
                frm = m.locations[i]
                to = m.locations[j]
                distance = math.sqrt((frm.x - to.x)**2 + (frm.y - to.y)**2)
                scaled_distance = int(distance * scale_factor)
                scaled_duration = int(DURATION_MATRIX[i][j])
                m.add_edge(frm, to, distance=scaled_distance, duration=scaled_duration)
        
        # 创建支持迁移的距离优化岛屿算法
        data = m.data()
        algo, rng = create_island_genetic_algorithm(
            data=data,
            seed=island_seed,
            hybrid_params=island_distance_params,  # 使用岛屿专属的距离优化配置
            custom_neighbourhood_params=NeighbourhoodParams(),  # 默认参数，实际由多样化配置生成器处理
            custom_swap_star_params=0.2,  # 保留用于后备，实际由多样化配置生成器处理
            island_id=island_id,
            migration_queue=migration_queue,
            is_distance_optimization=True  # 距离优化阶段
        )
        
        # 设置距离优化的迁移间隔和隔离期
        algo.migration_interval = 800  # 距离优化阶段更少的迁移
        algo.min_isolation_period = 1500  # 距离优化阶段更长的隔离期
        
        # 运行算法 - 距离优化阶段使用更长的延迟时间
        stop_criteria = EarlyStopAfterFeasible(
            max_runtime=runtime_limit, 
            delay_after_feasible=30  # 距离优化阶段延迟30秒
        )
        res = algo.run(stop=stop_criteria, collect_stats=False, display=False)
        
        if res.best.is_feasible():
            best_cost = res.best.distance() / scale_factor
            
            # 序列化解决方案数据
            routes_data = []
            for route in res.best.routes():
                route_visits = list(route.visits())
                routes_data.append(route_visits)
            
            solution_data = {
                'routes': routes_data,
                'distance': best_cost,
                'is_feasible': True,
                'num_vehicles': len(res.best.routes())
            }
            
            print(f"[距离优化岛屿 {island_id}] 完成，找到可行解，成本: {best_cost:.2f}")
            result_queue.put((island_id, best_cost, solution_data))
            return (island_id, best_cost, solution_data)
        else:
            print(f"[距离优化岛屿 {island_id}] 完成，未找到可行解")
            return None
            
    except Exception as e:
        print(f"[距离优化岛屿 {island_id}] 错误: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_route_details(route, coords, time_windows, service_times, scale_factor=SCALE_FACTOR):
    """
    计算路线的详细信息，包括距离、时间等，处理浮点精度
    
    参数:
    - route: 路线对象
    - coords: 坐标列表
    - time_windows: 时间窗列表
    - service_times: 服务时间列表
    - scale_factor: 距离和时间缩放因子，默认为全局缩放因子
    
    返回:
    - dict: 包含路线详细信息的字典
    """
    route_nodes = [0] + list(route.visits()) + [0]  # 添加起点和终点(仓库)
    
    # 计算准确的欧几里得距离
    total_distance = 0
    for i in range(len(route_nodes) - 1):
        from_idx = route_nodes[i]
        to_idx = route_nodes[i+1]
        dx = coords[from_idx][0] - coords[to_idx][0]
        dy = coords[from_idx][1] - coords[to_idx][1]
        distance = math.sqrt(dx*dx + dy*dy)
        total_distance += distance
    
    # 计算总时间（行驶时间 + 服务时间 + 等待时间）
    current_time = 0
    total_time = 0
    total_wait_time = 0
    total_service_time = 0
    
    for i in range(len(route_nodes) - 1):
        from_idx = route_nodes[i]
        to_idx = route_nodes[i+1]
        
        # 计算行驶时间（假设速度为1，即行驶时间等于距离）
        dx = coords[from_idx][0] - coords[to_idx][0]
        dy = coords[from_idx][1] - coords[to_idx][1]
        travel_time = math.sqrt(dx*dx + dy*dy)
        
        # 更新当前时间
        current_time += travel_time
        
        
        if i < len(route_nodes) - 1:  
            earliest_time = time_windows[to_idx][0]
            latest_time = time_windows[to_idx][1]
            
            if current_time < earliest_time:
                wait_time = earliest_time - current_time
                total_wait_time += wait_time
                current_time = earliest_time
            
            # 服务时间
            service_time = service_times[to_idx]
            total_service_time += service_time
            current_time += service_time
    
    total_time = total_distance + total_service_time + total_wait_time
    
    # 生成路线表示
    route_representation = " -> ".join([str(node) for node in route_nodes])
    
    return {
        "distance": total_distance,
        "time": total_time,
        "route": route_representation,
        "service_time": total_service_time,
        "wait_time": total_wait_time
    }

def get_min_vehicle_number(num_vehicles: int,
                          vehicle_capacity: int,
                          coords,
                          demands,
                          time_windows,
                          service_times):
    DURATION_MATRIX = [[math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) for c2 in coords] for c1 in coords]

    temp_model = Model()
    temp_model.add_vehicle_type(num_vehicles, capacity=vehicle_capacity, tw_early=time_windows[0][0], tw_late=time_windows[0][1])
    temp_model.add_depot(x=coords[0][0], y=coords[0][1], tw_early=time_windows[0][0], tw_late=time_windows[0][1])
    
    # 添加客户
    clients = [temp_model.add_client(
        x=coords[i][0], 
        y=coords[i][1], 
        tw_early=time_windows[i][0], 
        tw_late=time_windows[i][1], 
        delivery=demands[i], 
        service_duration=service_times[i]
    ) for i in range(1, len(coords))]
    
    # 添加边
    for i in range(len(coords)):
        for j in range(len(coords)):
            frm = temp_model.locations[i]
            to = temp_model.locations[j]
            distance = math.sqrt((frm.x - to.x)**2 + (frm.y - to.y)**2)
            scaled_distance = int(distance)
            scaled_duration = int(DURATION_MATRIX[i][j])
            temp_model.add_edge(frm, to, distance=scaled_distance, duration=scaled_duration)

    lower_bound = _lower_bound(temp_model.data())
    return lower_bound

def process_file(filepath, start_time=None, total_time_limit=1800):
    """
    使用混合策略处理单个数据文件
    
    参数:
    - filepath: 数据文件路径
    - start_time: 程序开始时间
    - total_time_limit: 总时间限制（秒），默认30分钟
    """
    filename = os.path.basename(filepath)
    try:
        # 使用全局缩放因子
        scale_factor = SCALE_FACTOR
        
        INITIAL_NUM_VEHICLES, VEHICLE_CAPACITY, COORDS, DEMANDS, TIME_WINDOWS, SERVICE_TIMES = load_solomon_data(filepath)

        best_solution_found = None
        # 设置初始尝试的车辆数
        num_vehicles_to_try = INITIAL_NUM_VEHICLES
        print(f"[{filename}] 开始搜索最小车辆数，起始车辆数: {num_vehicles_to_try}")

        # 时间控制变量
        if start_time is None:
            start_time = time.time()
        
        first_attempt = True
        vehicle_optimization_start = time.time()

        lower_bound = get_min_vehicle_number(num_vehicles_to_try, VEHICLE_CAPACITY, COORDS, DEMANDS, TIME_WINDOWS, SERVICE_TIMES)
        
        print(f"[{filename}] 计算理论下界: {lower_bound}")

        while num_vehicles_to_try >= lower_bound:
            # 检查总时间限制
            elapsed_total = time.time() - start_time
            if elapsed_total >= total_time_limit:
                print(f"[{filename}] 达到总时间限制 {total_time_limit/60:.1f} 分钟，停止车辆优化")
                break
                
            print(f"[{filename}] 正在尝试使用 {num_vehicles_to_try} 辆车...")
            
            # 确定运行时间
            if first_attempt:
                # 首次尝试使用600秒
                runtime_limit = 600
                first_attempt = False
                print(f"[{filename}] 首次尝试，使用 {runtime_limit} 秒运行时间")
            else:
                # 后续尝试根据剩余时间调整，但不超过400秒
                remaining_time = total_time_limit - elapsed_total
                runtime_limit = min(400, remaining_time * 0.1)  # 使用剩余时间的30%，但不超过400秒
                runtime_limit = max(60, runtime_limit)  # 至少60秒
                print(f"[{filename}] 后续尝试，使用 {runtime_limit:.0f} 秒运行时间")
            
            # 如果剩余时间不足，停止尝试更小的车辆数
            if elapsed_total + runtime_limit > total_time_limit * 0.8:  # 保留20%时间用于距离优化
                print(f"[{filename}] 剩余时间不足，停止尝试更小的车辆数")
                break 
            
            # 使用岛屿模型并行求解
            solution_result = solve_with_island_model(
                filepath=filepath,
                num_vehicles=num_vehicles_to_try,
                runtime_limit=runtime_limit,
                seed_base=41
            )
            
            if solution_result and solution_result['is_feasible']:
                print(f"[{filename}] 岛屿模型成功找到可行解。使用 {solution_result['num_vehicles']} 辆车，总距离: {solution_result['distance']:.2f}")
                
                # 创建一个模拟的solution对象来保持兼容性
                class MockSolution:
                    def __init__(self, solution_data, scale_factor):
                        self._solution_data = solution_data
                        self._scale_factor = scale_factor
                    
                    def routes(self):
                        # 创建模拟的路由对象
                        mock_routes = []
                        for route_visits in self._solution_data['routes']:
                            mock_route = MockRoute(route_visits)
                            mock_routes.append(mock_route)
                        return mock_routes
                    
                    def distance(self):
                        return self._solution_data['distance'] * self._scale_factor
                    
                    def is_feasible(self):
                        return self._solution_data['is_feasible']
                
                class MockRoute:
                    def __init__(self, visits):
                        self._visits = visits
                    
                    def visits(self):
                        return self._visits
                    
                    def delivery(self):
                        # 返回路径上所有客户的需求总和
                        total_demand = 0
                        for client_id in self._visits:
                            if client_id > 0:  # 排除仓库
                                total_demand += DEMANDS[client_id]
                        return [total_demand]  # 返回列表以与原接口兼容
                    
                    def time_warp(self):
                        return 0  # 假设无时间窗冲突
                
                best_solution_found = MockSolution(solution_result, scale_factor)
                actual_vehicles_used = len(solution_result['routes'])
                actual_distance = solution_result['distance']
                
                num_vehicles_to_try = actual_vehicles_used - 1
            else:
                print(f"[{filename}] 岛屿模型使用 {num_vehicles_to_try} 辆车未找到可行解。搜索结束。")
                break

        # 处理最终结果
        if best_solution_found:
            print(f"[{filename}] 最终确定最小车辆数为: {len(best_solution_found.routes())}")
            
            # 进一步优化最小车辆数的距离
            print(f"[{filename}] 进行额外的距离优化...")
            min_vehicles = len(best_solution_found.routes())
            
            # 准备精英解用于距离优化阶段
            elite_solutions = []
            if hasattr(best_solution_found, 'routes'):
                # 从车辆优化阶段提取精英解
                routes_data = []
                for route in best_solution_found.routes():
                    route_visits = list(route.visits())
                    routes_data.append(route_visits)
                
                elite_solution = {
                    'island_id': -1,  # 标记为来自车辆优化阶段
                    'cost': best_solution_found.distance() / scale_factor,
                    'solution': {
                        'routes': routes_data,
                        'num_vehicles': min_vehicles
                    }
                }
                elite_solutions.append(elite_solution)
            
            # 计算距离优化的运行时间 - 使用剩余的所有时间
            elapsed_total = time.time() - start_time
            remaining_time = total_time_limit - elapsed_total
            distance_opt_runtime = max(60, remaining_time - 10)  # 至少60秒，预留10秒缓冲时间
            
            print(f"[{filename}] 车辆优化完成，剩余时间 {remaining_time:.0f} 秒，距离优化将使用 {distance_opt_runtime:.0f} 秒")
            
            if distance_opt_runtime <= 10:
                print(f"[{filename}] 剩余时间不足，跳过距离优化阶段")
            else:
                distance_optimized_result = run_distance_optimization_with_islands(
                    filepath=filepath,
                    min_vehicles=min_vehicles,
                    runtime_limit=distance_opt_runtime,
                    seed_base=123,  # 使用不同的种子基数
                    elite_solutions=elite_solutions
                )
                
                if distance_optimized_result and distance_optimized_result['is_feasible']:
                    new_actual_distance = distance_optimized_result['distance']
                    previous_actual_distance = best_solution_found.distance() / scale_factor
                    
                    if new_actual_distance < previous_actual_distance:
                        # 创建新的MockSolution对象
                        class DistanceOptimizedSolution:
                            def __init__(self, solution_data, scale_factor):
                                self._solution_data = solution_data
                                self._scale_factor = scale_factor
                            
                            def routes(self):
                                mock_routes = []
                                for route_visits in self._solution_data['routes']:
                                    mock_route = MockRoute(route_visits)
                                    mock_routes.append(mock_route)
                                return mock_routes
                            
                            def distance(self):
                                return self._solution_data['distance'] * self._scale_factor
                            
                            def is_feasible(self):
                                return self._solution_data['is_feasible']
                        
                        class MockRoute:
                            def __init__(self, visits):
                                self._visits = visits
                            
                            def visits(self):
                                return self._visits
                            
                            def delivery(self):
                                total_demand = 0
                                for client_id in self._visits:
                                    if client_id > 0:
                                        total_demand += DEMANDS[client_id]
                                return [total_demand]
                            
                            def time_warp(self):
                                return 0
                        
                        best_solution_found = DistanceOptimizedSolution(distance_optimized_result, scale_factor)
                        print(f"[{filename}] 距离优化岛屿模型成功！新的总距离: {new_actual_distance:.2f}")
                    else:
                        print(f"[{filename}] 距离优化岛屿模型没有提高。保持原来的解决方案。")
                else:
                    print(f"[{filename}] 距离优化岛屿模型未找到可行解，保持原来的解决方案。")
            
            is_valid = validate_pyvrp_solution(
                solution=best_solution_found,
                num_all_clients=len(COORDS) - 1,
                vehicle_capacity=VEHICLE_CAPACITY,
                filename=filename,
                scale_factor=scale_factor # 传递缩放因子
            )
            if is_valid:
                # 重新计算每条路线的详细信息 - 这里计算的是真实距离，而非缩放后的距离
                routes = best_solution_found.routes()
                all_routes_details = []
                total_euclidean_distance = 0
                total_time = 0
                routes_text = []
                
                for route_idx, route in enumerate(routes):
                    details = calculate_route_details(
                        route, 
                        COORDS, 
                        TIME_WINDOWS, # 使用原始时间窗，因为calculate_route_details自己计算距离和时间
                        SERVICE_TIMES, # 使用原始服务时间
                        scale_factor=scale_factor # 传递缩放因子
                    )
                    total_euclidean_distance += details["distance"]
                    total_time += details["time"]
                    routes_text.append(f"路线{route_idx+1}: {details['route']}")
                    all_routes_details.append({
                        "route_idx": route_idx + 1,
                        "distance": details["distance"],
                        "time": details["time"],
                        "service_time": details["service_time"],
                        "wait_time": details["wait_time"],
                        "route": details["route"]
                    })
                
                # 将每条路线单独作为一列添加到结果中
                algorithm_name = f"混合MAB-Tournament策略(岛屿模型-{ISLAND_CONFIG['num_islands']}岛屿)"
                
                result_dict = {
                    "文件名": filename,
                    "Number of vehicles used": len(best_solution_found.routes()),
                    "Total distance (Euclidean)": f"{total_euclidean_distance:.2f}",
                    "Total time": f"{total_time:.2f}",
                    "Algorithm": algorithm_name
                }
                
                # 为每条路线添加单独的列
                for i, route_detail in enumerate(all_routes_details):
                    result_dict[f"Route_{i+1}"] = route_detail["route"]
                
                return result_dict
            else: 
                return {"文件名": filename, "Number of vehicles used": "Validation Failed",  "Total distance (Euclidean)": "N/A", "Total time": "N/A"}
        else:
            print(f"[{filename}] 即使用初始的 {INITIAL_NUM_VEHICLES} 辆车也未找到任何可行解。")
            return {"文件名": filename, "Number of vehicles used": "No Feasible Solution",  "Total distance (Euclidean)": "N/A", "Total time": "N/A"}
    except Exception as e:
        print(f"处理文件 {filepath} 时发生严重错误: {e}")
        return {"文件名": filename, "Number of vehicles used": "Processing Error", "Total distance (Euclidean)": str(e), "Total time": "N/A"}

def main():
    try:
        # 程序开始时启动计时 - 每个文件30分钟
        start_time = time.time()
        
        print(f"开始执行，每个文件处理时间限制: 30 分钟")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        directories_names = ['h400']
        directories_paths = [os.path.join(script_dir, name) for name in directories_names]
        
        print(f"使用MAB与锦标赛混合选择策略+岛屿模型({ISLAND_CONFIG['num_islands']}岛屿)运行实验")
        
        all_filepaths = []
        print("正在搜集需要处理的文件...")
        for directory_path in directories_paths:
            if not os.path.isdir(directory_path):
                print(f"警告: 目录 '{directory_path}' 不存在，已跳过。")
                continue
            for filename in os.listdir(directory_path):
                if filename.lower().endswith(('.txt', '.vrp')):
                    filepath = os.path.join(directory_path, filename)
                    all_filepaths.append(filepath)

        if not all_filepaths:
            print("未找到任何 .txt 或 .vrp 文件进行处理。")
            return

        # 顺序处理所有文件
        all_results = []
        if all_filepaths:
            print(f"共找到 {len(all_filepaths)} 个文件。将顺序处理所有文件。")
            
            # 遍历所有文件并顺序处理
            for i, filepath in enumerate(all_filepaths):
                filename = os.path.basename(filepath)
                print(f"\n正在处理第 {i+1}/{len(all_filepaths)} 个文件: {filename}")
                
                # 每个文件都有30分钟的处理时间
                file_time_limit = 30 * 60  # 30分钟转换为秒
                file_start_time = time.time()  # 每个文件的开始时间
                
                print(f"为文件 {filename} 分配时间限制: {file_time_limit/60:.0f} 分钟")
                
                # 处理单个文件，每个文件独立计时
                result = process_file(filepath, start_time=file_start_time, total_time_limit=file_time_limit)
                if result:
                    all_results.append(result)
                
                # 计算该文件的实际处理时间
                file_elapsed_time = time.time() - file_start_time
                print(f"文件 {filename} 处理完成，用时: {file_elapsed_time/60:.1f} 分钟，已处理 {len(all_results)}/{i+1} 个文件")

        if all_results:
            df = pd.DataFrame(all_results)
            df = df.sort_values(by="文件名").reset_index(drop=True)
            
            # 输出文件
            output_csv_file = os.path.join(script_dir, f"200_{ISLAND_CONFIG['num_islands']}_parallel.csv")
            df.to_csv(output_csv_file, index=False)

            print(f"\n\n批量处理完成！结果已保存到 {output_csv_file}")
            # 只打印文件名和距离列
            print(df[["文件名", "Total distance (Euclidean)"]])
        else:
            print("\n\n未能处理任何文件，没有生成结果。")
    except Exception as e:
        print(f"主程序发生错误: {e}")
    finally:
        # 计算并打印程序运行时间
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'\n程序总执行时间: {execution_time/60:.1f} 分钟 ({execution_time:.0f} 秒)')
        print('进程已结束, 退出代码0')


if __name__ == "__main__":
    # Windows系统需要保护主模块入口
    multiprocessing.freeze_support()
    main()
import math
from typing import Tuple, Dict, List
from pyvrp._pyvrp import Solution, CostEvaluator, RandomNumberGenerator
from pyvrp.Population import Population
from pyvrp.GeneticAlgorithm import GeneticAlgorithm
import time
from pyvrp.ProgressPrinter import ProgressPrinter
from pyvrp.Result import Result
from pyvrp.Statistics import Statistics
from pyvrp.stop.StoppingCriterion import StoppingCriterion
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import queue
import pickle
import threading
import copy

class MABParentSelection:
    def __init__(
      self, num_arms: int = 5, c: float = 1.0, initial_temperature: float = 1.0, cooling_rate: float = 0.9999,
      custom_vehicle_weight: float = 5.0,
      custom_distance_weight: float = 1.0):
        self.num_arms = num_arms  
        self.c = c                
        self.pulls = {}           
        self.rewards = {}         
        self.total_pulls = 0      
        self.temperature = initial_temperature  # 初始温度参数
        self.cooling_rate = cooling_rate        # 温度降低系数
        self.best_routes_seen = float('inf')    # 记录到目前为止见过的最少车辆数
        self.no_improvement_count = 0           # 记录没有改进的迭代次数
        self.custom_vehicle_weight = custom_vehicle_weight
        self.custom_distance_weight = custom_distance_weight

    def select(self, population: Population, rng: RandomNumberGenerator, 
               cost_evaluator: CostEvaluator) -> Tuple[Solution, Solution]:
       
        # 更新温度
        self.temperature *= self.cooling_rate
        
        population._update_fitness(cost_evaluator)
        
        # 选择第一个父代
        first_parent = self._select_one_parent(population, rng)
        
        # 选择与第一个父代不同的第二个父代
        second_parent = first_parent
        max_attempts = 10  # 最大尝试次数，避免无限循环
        attempt = 0
        
        while id(second_parent) == id(first_parent) and attempt < max_attempts:
            second_parent = self._select_one_parent(population, rng)
            attempt += 1
        
        return first_parent, second_parent
    
    def _select_one_parent(self, population: Population, rng: RandomNumberGenerator) -> Solution:
        # 种群大小
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("种群为空，无法选择父代")
        
        # 从种群中随机选择num_arms个解
        indices = [rng.randint(pop_size) for _ in range(min(self.num_arms, pop_size))]
        
        # 计算每个解的UCB值并选择最高的
        best_solution = None
        best_ucb = float('-inf')
        
        for idx in indices:
            # 获取对应索引的解
            if idx < population.num_feasible():
                solution = population._feas[idx].solution
            else:
                solution = population._infeas[idx - population.num_feasible()].solution
            
            # 解的标识符（使用内存地址作为唯一标识）
            solution_id = id(solution)
            
            # 初始化该解的数据（如果之前未见过）
            if solution_id not in self.pulls:
                self.pulls[solution_id] = 0
                self.rewards[solution_id] = 0.0
            
            # 计算UCB值
            if self.pulls[solution_id] == 0:
                # 未被选择过的解优先选择
                ucb = float('inf')
            else:
                # UCB公式: 平均奖励 + c * sqrt(ln(总次数) / 该臂次数)
                exploitation = self.rewards[solution_id] / self.pulls[solution_id]
                
                # 考虑温度因素：温度越低，探索项越小
                effective_c = self.c * self.temperature
                exploration = effective_c * math.sqrt(math.log(self.total_pulls + 1) / self.pulls[solution_id])
                
                ucb = exploitation + exploration
            
            # 更新最佳UCB值
            if ucb > best_ucb:
                best_ucb = ucb
                best_solution = solution
        
        # 更新选中解的统计信息
        if best_solution:
            solution_id = id(best_solution)
            self.pulls[solution_id] += 1
            self.total_pulls += 1
        
        return best_solution
    
    def update_reward(self, solution: Solution, offspring: Solution, cost_evaluator: CostEvaluator):
        solution_id = id(solution)
        
        # 如果该解不在记录中，则忽略
        if solution_id not in self.rewards:
            return
        
        # 获取父代和子代的信息
        parent_routes = len(solution.routes())
        parent_distance = solution.distance()
        parent_cost = cost_evaluator.penalised_cost(solution)
        parent_feasible = solution.is_feasible()
        
        offspring_routes = len(offspring.routes())
        offspring_distance = offspring.distance()
        offspring_cost = cost_evaluator.penalised_cost(offspring)
        offspring_feasible = offspring.is_feasible()
        
        # 更新见过的最佳车辆数
        if offspring_feasible and offspring_routes < self.best_routes_seen:
            self.best_routes_seen = offspring_routes
            self.no_improvement_count = 0  # 重置无改进计数
        else:
            self.no_improvement_count += 1
        
        # 初始化奖励值
        reward = 0.0
        
        # 动态调整权重参数（随着搜索的进行，可能希望更加关注最小化距离）
        vehicle_weight = self.custom_vehicle_weight  # 基础车辆数量权重
        # 如果长时间没有改进，适当减小车辆权重，更多关注距离
        if self.no_improvement_count > 10000:
            vehicle_weight *= 0.8
        
        distance_weight = self.custom_distance_weight # 距离权重
        
        # 优先考虑车辆数量的减少
        if offspring_routes < parent_routes:
            # 减少车辆数是最高优先级，给予高奖励
            routes_improvement = (parent_routes - offspring_routes) / parent_routes
            reward += vehicle_weight * (1.0 + routes_improvement * 2)
            
            # 额外奖励：解创建了新的车辆数记录
            if offspring_feasible and offspring_routes == self.best_routes_seen:
                reward += 2.0
        elif offspring_routes == parent_routes:
            # 车辆数相同，考虑距离改进
            if offspring_distance < parent_distance:
                # 距离改进，给予中等奖励
                distance_improvement = (parent_distance - offspring_distance) / parent_distance
                # 根据改进程度给予不同的奖励
                if distance_improvement > 0.1:  # 显著改进
                    reward += distance_weight * (0.8 + distance_improvement)
                else:  # 小幅改进
                    reward += distance_weight * (0.5 + distance_improvement)
            else:
                # 距离没有改进，但成本可能有改进（例如减少了时间窗违反等）
                if offspring_cost < parent_cost:
                    cost_improvement = (parent_cost - offspring_cost) / parent_cost
                    reward += 0.2 * cost_improvement
                else:
                    # 没有任何改进，给予小惩罚
                    reward += 0.05
        else:
            if offspring_cost < parent_cost:
                cost_improvement = (parent_cost - offspring_cost) / parent_cost
                # 只有当成本改善显著时才给予一定的奖励
                if cost_improvement > 0.2:
                    reward += 0.15 * cost_improvement
                else:
                    reward += 0.05 * cost_improvement
            else:
                # 车辆数增加且没有其他改进，给予明显惩罚
                reward += 0.01
        
        # 可行性奖励
        if offspring_feasible and not parent_feasible:
            # 从不可行变为可行，给予高奖励
            reward += 1.5
        elif not offspring_feasible and parent_feasible:
            # 从可行变为不可行，给予高惩罚
            reward -= 1.0
        
        # 多样性奖励：如果该解与大多数其他解不同，可能包含有价值的多样性
        # 这里使用简单的启发式方法：车辆数与当前最佳车辆数的差距
        if offspring_feasible:
            routes_gap = offspring_routes - self.best_routes_seen
            if routes_gap <= 2:  # 接近最佳车辆数的解可能更有价值
                reward += 0.2 * (3 - routes_gap)  # 差距越小，奖励越高
        
        # 更新累计奖励
        self.rewards[solution_id] += reward


class TournamentSelection:
    """
    实现基于锦标赛的父代选择策略
    """
    def __init__(self, tournament_size: int = 5):
        self.tournament_size = tournament_size
        
    def select(self, population: Population, rng: RandomNumberGenerator, 
               cost_evaluator: CostEvaluator) -> Tuple[Solution, Solution]:
        """
        使用锦标赛选择策略选择两个父代
        """
        population._update_fitness(cost_evaluator)
        
        # 选择第一个父代
        first_parent = self._select_one_parent(population, rng)
        
        # 选择与第一个父代不同的第二个父代
        second_parent = first_parent
        max_attempts = 10
        attempt = 0
        
        while id(second_parent) == id(first_parent) and attempt < max_attempts:
            second_parent = self._select_one_parent(population, rng)
            attempt += 1
            
        return first_parent, second_parent
    
    def _select_one_parent(self, population: Population, rng: RandomNumberGenerator) -> Solution:
        """
        通过锦标赛选择单个解
        """
        pop_size = len(population)
        if pop_size == 0:
            raise ValueError("种群为空，无法选择父代")
            
        # 从种群中随机选择tournament_size个解
        competitors = []
        for _ in range(min(self.tournament_size, pop_size)):
            idx = rng.randint(pop_size)
            if idx < population.num_feasible():
                solution = population._feas[idx].solution
                fitness = population._feas[idx].fitness
            else:
                solution = population._infeas[idx - population.num_feasible()].solution
                fitness = population._infeas[idx - population.num_feasible()].fitness
            competitors.append((solution, fitness))
        
        # 选择适应度最高的解
        best_solution, best_fitness = max(competitors, key=lambda x: x[1])
        return best_solution
    
    def update_reward(self, solution: Solution, offspring: Solution, cost_evaluator: CostEvaluator):
        """
        锦标赛选择不需要更新奖励，此方法为了兼容MAB接口
        """
        pass


class HybridSelection:
    """
    结合MAB和锦标赛选择的混合策略
    """
    def __init__(
        self, 
        switch_interval: int = 100,  # 多少次迭代后评估和切换策略
        tournament_size: int = 5,
        mab_num_arms: int = 8,
        mab_c: float = 1.2,
        initial_temperature: float = 1.5,
        cooling_rate: float = 0.9998,
        custom_vehicle_weight: float = 5.0,
        custom_distance_weight: float = 0.0
    ):
        self.mab_selector = MABParentSelection(
            num_arms=mab_num_arms,
            c=mab_c,
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            custom_vehicle_weight=custom_vehicle_weight,
            custom_distance_weight=custom_distance_weight
        )
        self.tournament_selector = TournamentSelection(tournament_size=tournament_size)
        
        self.switch_interval = switch_interval
        self.iterations = 0
        self.current_strategy = "mab"  # 初始使用MAB策略
        
        # 用于跟踪策略性能的指标
        self.mab_performance = []
        self.tournament_performance = []
        self.strategy_history = []
        
        # 记录最近一次使用每种策略的最佳解
        self.last_mab_best = None
        self.last_tournament_best = None
        
        # 当前使用策略的性能指标
        self.current_best_cost = float('inf')
        self.iterations_without_improvement = 0
        
    def select(self, population: Population, rng: RandomNumberGenerator, 
               cost_evaluator: CostEvaluator) -> Tuple[Solution, Solution]:
        """
        根据当前选择的策略选择父代
        """
        self.iterations += 1
        
        # 获取当前最佳解的成本
        current_best_cost = self._get_best_cost(population, cost_evaluator)
        
        # 记录性能指标
        if current_best_cost < self.current_best_cost:
            self.current_best_cost = current_best_cost
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
            
        # 每隔switch_interval次迭代评估并可能切换策略
        if self.iterations % self.switch_interval == 0:
            self._evaluate_and_switch_strategy(population, cost_evaluator)
            
        # 使用当前策略选择父代
        if self.current_strategy == "mab":
            return self.mab_selector.select(population, rng, cost_evaluator)
        else:
            return self.tournament_selector.select(population, rng, cost_evaluator)
    
    def _get_best_cost(self, population: Population, cost_evaluator: CostEvaluator) -> float:
        """
        获取种群中最佳解的成本
        """
        best_cost = float('inf')
        
        # 检查可行解
        for i in range(population.num_feasible()):
            sol = population._feas[i].solution
            cost = cost_evaluator.cost(sol)
            if cost < best_cost:
                best_cost = cost
                
        # 如果没有可行解，则考虑不可行解
        if best_cost == float('inf') and len(population) > 0:
            for i in range(population.num_feasible(), len(population)):
                idx = i - population.num_feasible()
                sol = population._infeas[idx].solution
                cost = cost_evaluator.penalised_cost(sol)  # 对于不可行解使用惩罚成本
                if cost < best_cost:
                    best_cost = cost
        
        return best_cost
    
    def _get_best_solution(self, population: Population, cost_evaluator: CostEvaluator) -> Tuple[Solution, float, int]:
        """
        获取种群中的最佳解、成本和路线数
        
        Returns:
            Tuple[Solution, float, int]: (最佳解, 成本, 路线数)，如果没有找到解则返回(None, inf, inf)
        """
        best_solution = None
        best_cost = float('inf')
        best_routes = float('inf')
        
        # 先检查可行解
        for i in range(population.num_feasible()):
            sol = population._feas[i].solution
            cost = cost_evaluator.cost(sol)
            if cost < best_cost:
                best_cost = cost
                best_solution = sol
                best_routes = len(sol.routes())
        
        # 如果没有可行解，则考虑不可行解
        if best_solution is None and len(population) > population.num_feasible():
            for i in range(population.num_feasible(), len(population)):
                idx = i - population.num_feasible()
                sol = population._infeas[idx].solution
                cost = cost_evaluator.penalised_cost(sol)  # 对于不可行解使用惩罚成本
                if cost < best_cost:
                    best_cost = cost
                    best_solution = sol
                    best_routes = float('inf')  # 不可行解的路线数视为无限
        
        return best_solution, best_cost, best_routes
    
    def _evaluate_and_switch_strategy(self, population: Population, cost_evaluator: CostEvaluator):
        """
        评估当前策略的性能并决定是否切换
        """
        best_solution, best_cost, best_routes = self._get_best_solution(population, cost_evaluator)
        
        # 如果没有找到解，直接返回
        if best_solution is None:
            return
        
        # 记录当前策略的性能
        performance = {
            'cost': best_cost,
            'routes': best_routes,
            'iterations_without_improvement': self.iterations_without_improvement
        }
        
        if self.current_strategy == "mab":
            self.mab_performance.append(performance)
            self.last_mab_best = performance
        else:
            self.tournament_performance.append(performance)
            self.last_tournament_best = performance
        
        # 决定是否切换策略
        should_switch = False
        
        # 如果两种策略都有记录，比较它们
        if self.last_mab_best and self.last_tournament_best:
            # 首先比较车辆数
            if self.last_mab_best['routes'] < self.last_tournament_best['routes']:
                should_switch = (self.current_strategy != "mab")
            elif self.last_tournament_best['routes'] < self.last_mab_best['routes']:
                should_switch = (self.current_strategy != "tournament")
            else:
                # 车辆数相同，比较成本
                if self.last_mab_best['cost'] < self.last_tournament_best['cost']:
                    should_switch = (self.current_strategy != "mab")
                elif self.last_tournament_best['cost'] < self.last_mab_best['cost']:
                    should_switch = (self.current_strategy != "tournament")
                else:
                    # 如果连续多次没有改进，尝试切换策略
                    if self.iterations_without_improvement > self.switch_interval // 2:
                        should_switch = True
        else:
            # 如果有一种策略没有记录，尝试使用另一种
            if not self.last_mab_best:
                should_switch = (self.current_strategy != "mab")
            elif not self.last_tournament_best:
                should_switch = (self.current_strategy != "tournament")
        
        # 如果当前策略长时间没有改进，也考虑切换
        if self.iterations_without_improvement > self.switch_interval * 2:
            should_switch = True
            
        # 执行切换
        if should_switch:
            self.current_strategy = "tournament" if self.current_strategy == "mab" else "mab"
            self.iterations_without_improvement = 0
            # print(f"切换选择策略到: {self.current_strategy}")
            
        # 记录策略历史
        self.strategy_history.append(self.current_strategy)
    
    def update_reward(self, solution: Solution, offspring: Solution, cost_evaluator: CostEvaluator):
        """
        更新MAB选择器中的奖励
        """
        if self.current_strategy == "mab":
            self.mab_selector.update_reward(solution, offspring, cost_evaluator)
        # 对于tournament策略，不需要更新奖励


class MABGeneticAlgorithm(GeneticAlgorithm):
    
    def __init__(self, *args, hybrid_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化混合选择器
        if hybrid_params is None:
            hybrid_params = {}
            
        self._selector = HybridSelection(
            switch_interval=hybrid_params.get('switch_interval', 100),
            tournament_size=hybrid_params.get('tournament_size', 5),
            mab_num_arms=hybrid_params.get('num_arms', 8),
            mab_c=hybrid_params.get('c', 1.2),
            initial_temperature=hybrid_params.get('initial_temperature', 1.5),
            cooling_rate=hybrid_params.get('cooling_rate', 0.9998),
            custom_vehicle_weight=hybrid_params.get('custom_vehicle_weight', 5.0),
            custom_distance_weight=hybrid_params.get('custom_distance_weight', 0.0)
        )
    
    def run(
        self,
        stop: StoppingCriterion,
        collect_stats: bool = True,
        display: bool = False,
        display_interval: float = 5.0,
    ):
        try:
            print_progress = ProgressPrinter(display, display_interval)
        except TypeError:
            print_progress = ProgressPrinter(display)
        
        print_progress.start(self._data)

        start = time.perf_counter()
        stats = Statistics(collect_stats=collect_stats)
        iters = 0
        iters_no_improvement = 1

        for sol in self._initial_solutions:
            self._pop.add(sol, self._cost_evaluator)

        # 为自定义停止条件设置种群引用
        if hasattr(stop, 'set_population_reference'):
            stop.set_population_reference(self._pop, self._cost_evaluator)

        restart_threshold = 20000  # 默认值
        try:
            restart_threshold = getattr(self._params, "num_iters_no_improvement", restart_threshold)
        except (AttributeError, TypeError):
            print("注意：使用默认重启阈值20000")
            
        while not stop(self._cost_evaluator.cost(self._best)):
            iters += 1

            if iters_no_improvement == restart_threshold:
                print_progress.restart()

                iters_no_improvement = 1
                self._pop.clear()

                for sol in self._initial_solutions:
                    self._pop.add(sol, self._cost_evaluator)

            curr_best = self._cost_evaluator.cost(self._best)

            # 使用混合选择策略选择父代
            parents = self._selector.select(self._pop, self._rng, self._cost_evaluator)
            offspring = self._crossover(
                parents, self._data, self._cost_evaluator, self._rng
            )
            
            # 改进子代
            self._improve_offspring(offspring)
            
            # 对父代进行奖励更新
            for parent in parents:
                self._selector.update_reward(parent, offspring, self._cost_evaluator)

            new_best = self._cost_evaluator.cost(self._best)

            if new_best < curr_best:
                iters_no_improvement = 1
            else:
                iters_no_improvement += 1

            stats.collect_from(self._pop, self._cost_evaluator)
            print_progress.iteration(stats)

        end = time.perf_counter() - start
        res = Result(self._best, stats, iters, end)

        print_progress.end(res)

        return res

from pyvrp.search import (
    NODE_OPERATORS,
    ROUTE_OPERATORS,
    LocalSearch,
    NeighbourhoodParams,
    compute_neighbours,
    SwapStar,
)

class DiverseIntensityLocalSearch:
    """
    多样化搜索强度的LocalSearch包装器
    
    通过控制搜索的调用次数和强度来实现不同岛屿间的搜索行为差异
    """
    
    def __init__(self, local_search, intensity_config):
        self.local_search = local_search
        self.intensity_factor = intensity_config['factor']
        self.intensity_name = intensity_config['name']
        self.call_count = 0
        
    def __call__(self, solution, cost_evaluator):
        """实现多样化的搜索强度策略"""
        improved_solution = solution
        
        if self.intensity_factor <= 0.5:
            # 轻度搜索：有概率跳过局部搜索
            import random
            if random.random() > 0.7:  # 30%概率进行搜索
                improved_solution = self.local_search(improved_solution, cost_evaluator)
        elif self.intensity_factor <= 1.0:
            # 标准搜索：正常调用一次
            improved_solution = self.local_search(improved_solution, cost_evaluator)
        elif self.intensity_factor <= 2.0:
            # 密集搜索：多次调用，每次都尝试改进
            for _ in range(int(self.intensity_factor)):
                new_solution = self.local_search(improved_solution, cost_evaluator)
                if cost_evaluator.cost(new_solution) < cost_evaluator.cost(improved_solution):
                    improved_solution = new_solution
                else:
                    break  # 如果没有改进就停止
        else:
            # 极限搜索：强制多次调用
            for _ in range(int(self.intensity_factor)):
                improved_solution = self.local_search(improved_solution, cost_evaluator)
        
        self.call_count += 1
        return improved_solution
    
    @property
    def node_operators(self):
        """委托node_operators属性给原始LocalSearch"""
        return self.local_search.node_operators
    
    @property
    def route_operators(self):
        """委托route_operators属性给原始LocalSearch"""
        return self.local_search.route_operators
    
    @property
    def neighbours(self):
        """委托neighbours属性给原始LocalSearch"""
        return self.local_search.neighbours
    
    @neighbours.setter
    def neighbours(self, value):
        """委托neighbours设置给原始LocalSearch"""
        self.local_search.neighbours = value
    
    def add_node_operator(self, op):
        """委托add_node_operator方法给原始LocalSearch"""
        return self.local_search.add_node_operator(op)
    
    def add_route_operator(self, op):
        """委托add_route_operator方法给原始LocalSearch"""
        return self.local_search.add_route_operator(op)
    
    def __getattr__(self, name):
        """委托其他属性和方法给原始LocalSearch"""
        return getattr(self.local_search, name)
# 用于创建并配置混合选择算法
def create_hybrid_genetic_algorithm(data, 
    seed=42, 
    hybrid_params=None, 
    custom_neighbourhood_params=NeighbourhoodParams(),
    custom_swap_star_params=0.05,
    island_id=None
):
    from pyvrp.solve import SolveParams
    from pyvrp.GeneticAlgorithm import GeneticAlgorithmParams
    from pyvrp.PenaltyManager import PenaltyManager
    from pyvrp.Population import Population
    from pyvrp._pyvrp import RandomNumberGenerator, Solution
    from pyvrp.diversity import broken_pairs_distance as bpd
    from pyvrp.crossover import selective_route_exchange as srex
    from pyvrp.crossover import ordered_crossover as ox
    from pyvrp.search import SwapStar
    
    genetic_params = GeneticAlgorithmParams()
    
    # 解算器参数
    params = SolveParams(
        genetic=genetic_params,
        neighbourhood=custom_neighbourhood_params,
    )
    
    # 初始化解算组件
    rng = RandomNumberGenerator(seed=seed)
    
    # 使用多样化的局部搜索（如果指定了岛屿ID）
    if island_id is not None:
        ls = DiverseSearchConfigGenerator.create_diverse_local_search(data, rng, island_id)
    else:
        # 默认使用标准配置
        neighbours = compute_neighbours(data, params.neighbourhood)
        ls = LocalSearch(data, rng, neighbours)
        
        # 使用与原始solve函数相同的方式添加搜索算子
        for node_op in NODE_OPERATORS:
            try:
                if hasattr(node_op, 'supports') and node_op.supports(data):
                    ls.add_node_operator(node_op(data))
                else:
                    ls.add_node_operator(node_op(data))
            except Exception as e:
                print(f"添加节点算子 {node_op.__name__} 时出错: {e}")
        
        # 特别处理ROUTE_OPERATORS，为SwapStar使用自定义参数
        for route_op in ROUTE_OPERATORS:
            try:
                # 检查是否是SwapStar算子
                if route_op.__name__ == 'SwapStar':
                    # 使用自定义参数创建SwapStar
                    swap_star_instance = SwapStar(data, overlap_tolerance=custom_swap_star_params)
                    ls.add_route_operator(swap_star_instance)
                else:
                    # 其他算子使用默认方式
                    if hasattr(route_op, 'supports') and route_op.supports(data):
                        ls.add_route_operator(route_op(data))
                    else:
                        ls.add_route_operator(route_op(data))
            except Exception as e:
                print(f"添加路径算子 {route_op.__name__} 时出错: {e}")
    
    pm = PenaltyManager.init_from(data, params.penalty)
    pop = Population(bpd, params.population)
    
    # 使用多样化的初始解生成策略（如果指定了岛屿ID）
    if island_id is not None:
        init = DiverseInitialSolutionGenerator.generate_diverse_initial_solutions(
            data, rng, params.population.min_pop_size, island_id
        )
    else:
        # 默认使用随机初始解
        init = [Solution.make_random(data, rng) for _ in range(params.population.min_pop_size)]
    
    crossover = srex if data.num_vehicles > 1 else ox
    
    # 初始化混合算法
    algo = MABGeneticAlgorithm(
        data, pm, rng, pop, ls, crossover, init, 
        params=genetic_params, hybrid_params=hybrid_params
    )
    
    return algo, rng 


class IslandGeneticAlgorithm(MABGeneticAlgorithm):
    """支持迁移机制的岛屿遗传算法"""
    
    def __init__(self, *args, island_id=None, migration_queue=None, is_distance_optimization=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.island_id = island_id
        self.migration_queue = migration_queue
        self.migration_counter = 0
        self.base_migration_interval = 500 if not is_distance_optimization else 800
        self.migration_interval = self.base_migration_interval
        self.min_isolation_period = 1000 if not is_distance_optimization else 1500  # 最小隔离期
        self.received_migrations = []
        self.is_distance_optimization = is_distance_optimization
        self.elite_solutions = []  # 存储精英解
        self.last_migration_time = 0
        self.stagnation_counter = 0  # 停滞计数器
        self.last_best_cost = float('inf')
        self.diversity_threshold = 0.1  # 多样性阈值
        self.migration_success_rate = 0.0  # 迁移成功率
        self.total_migrations_sent = 0
        self.successful_migrations = 0
        
    def process_migrations(self):
        """处理从其他岛屿迁移来的解决方案（增强版）"""
        try:
            migrations_processed = 0
            # 动态调整最大迁移数量：停滞时接受更多迁移
            base_max = 2 if not self.is_distance_optimization else 1
            max_migrations = base_max + min(self.stagnation_counter // 50, 3)
            
            accepted_migrations = 0
            
            while migrations_processed < max_migrations:
                try:
                    migration_data = self.migration_queue.get(timeout=0.001)  # 非阻塞获取
                    migrations_processed += 1
                    
                    # 跳过自己发送的迁移
                    if migration_data['island_id'] == self.island_id:
                        continue
                    
                    # 获取当前最优解信息
                    migration_cost = migration_data['cost']
                    current_best_cost = self._cost_evaluator.cost(self._best) if self._best else float('inf')
                    
                    # 多层次接受策略
                    should_accept = self._evaluate_migration_acceptance(
                        migration_data, migration_cost, current_best_cost
                    )
                    
                    if should_accept:
                        # 重建解决方案并添加到种群
                        routes_data = migration_data['solution']['routes']
                        if routes_data:
                            try:
                                # 创建Solution对象
                                migrated_solution = Solution(self._data, routes_data)
                                
                                # 添加到种群中（可能替换较差的解）
                                self._add_migrated_solution(migrated_solution, migration_data['island_id'])
                                accepted_migrations += 1
                                
                            except Exception as e:
                                print(f"[岛屿 {self.island_id}] 重建迁移解时出错: {e}")
                    
                except queue.Empty:
                    break
            
            # if accepted_migrations > 0:
            #     print(f"[岛屿 {self.island_id}] 接受了 {accepted_migrations}/{migrations_processed} 个迁移解")
                    
        except Exception as e:
            print(f"[岛屿 {self.island_id}] 处理迁移时出错: {e}")
    
    def _evaluate_migration_acceptance(self, migration_data, migration_cost, current_best_cost):
        """评估是否接受迁移解的更严格策略"""
        migration_vehicles = migration_data['solution'].get('num_vehicles', float('inf'))
        current_vehicles = len(self._best.routes()) if self._best else float('inf')
        
        # 层次1：只接受明显更优的解
        if not self.is_distance_optimization:
            # 车辆优化阶段 - 更严格的车辆数要求
            if migration_vehicles < current_vehicles:
                return True
            elif migration_vehicles == current_vehicles and migration_cost < current_best_cost * 0.95:  # 保持原有严格度
                return True
        else:
            # 距离优化阶段 - 更严格的距离要求
            cost_improvement = (current_best_cost - migration_cost) / current_best_cost if current_best_cost > 0 else 0
            if cost_improvement > 0.05:  # 提高改进要求从2%到5%
                return True
        
        # 层次2：长期停滞时的有限接受（门槛更高）
        if self.stagnation_counter > 1000:  # 停滞阈值从100提高到1000
            relaxed_threshold = 0.95 if not self.is_distance_optimization else 0.98  # 提高门槛
            if migration_cost < current_best_cost * relaxed_threshold:
                return True
        
        # 层次3：极端停滞时的多样性接受（门槛依然很高）
        if self.stagnation_counter > 3000:  # 极端停滞阈值
            if self._is_population_diverse() < self.diversity_threshold:
                # 只有在多样性很低时才接受稍差的解
                diversity_threshold = 1.02 if not self.is_distance_optimization else 1.01  # 降低接受度
                if migration_cost < current_best_cost * diversity_threshold:
                    return True
        
        return False
    
    def _add_migrated_solution(self, migrated_solution, source_island_id):
        """智能地将迁移解添加到种群中"""
        if migrated_solution.is_feasible():
            # 检查是否为新的最优解
            if self._cost_evaluator.cost(migrated_solution) < self._cost_evaluator.cost(self._best):
                self._best = migrated_solution
                self.successful_migrations += 1
                # print(f"[岛屿 {self.island_id}] 接受来自岛屿 {source_island_id} 的最优解")
            
            # 添加到种群（PyVRP会自动管理种群大小）
            self._pop.add(migrated_solution, self._cost_evaluator)
        else:
            # 不可行解也可能有价值，添加到不可行种群
            self._pop.add(migrated_solution, self._cost_evaluator)
    
    def _is_population_diverse(self):
        """评估种群多样性（简化版）"""
        if len(self._pop) < 2:
            return 1.0
        
        # 计算可行解中车辆数的方差作为多样性指标
        vehicle_counts = []
        for i in range(min(10, self._pop.num_feasible())):
            sol = self._pop._feas[i].solution
            vehicle_counts.append(len(sol.routes()))
        
        if len(vehicle_counts) < 2:
            return 0.0
        
        mean_vehicles = sum(vehicle_counts) / len(vehicle_counts)
        variance = sum((x - mean_vehicles) ** 2 for x in vehicle_counts) / len(vehicle_counts)
        
        # 归一化多样性值
        return min(variance / max(mean_vehicles, 1), 1.0)
    
    def send_migration(self):
        """发送当前最优解和多样性解供其他岛屿使用（增强版）"""
        try:
            if self._best and self._best.is_feasible():
                current_cost = self._best.distance()
                
                # 初始化发送成本记录
                if not hasattr(self, 'last_sent_cost'):
                    self.last_sent_cost = float('inf')
                
                # 动态调整发送阈值：停滞时更积极发送
                base_threshold = 0.01 if not self.is_distance_optimization else 0.005
                improvement_threshold = base_threshold * max(0.1, 1 - self.stagnation_counter / 1000)
                
                cost_improvement = (self.last_sent_cost - current_cost) / self.last_sent_cost if self.last_sent_cost > 0 else 1.0
                
                should_send = (
                    cost_improvement > improvement_threshold or  # 显著改进
                    self.stagnation_counter > 200 or  # 长期停滞
                    self.total_migrations_sent == 0  # 首次发送
                )
                
                if should_send:
                    # 发送最优解
                    self._send_best_solution(current_cost)
                    
                    # 额外发送多样性解（如果种群足够大）
                    if self._pop.num_feasible() > 5 and self.stagnation_counter > 100:
                        self._send_diversity_solutions()
                    
        except Exception as e:
            print(f"[岛屿 {self.island_id}] 发送迁移时出错: {e}")
    
    def _send_best_solution(self, current_cost):
        """发送最优解"""
        try:
            # 序列化当前最优解
            routes_data = []
            for route in self._best.routes():
                route_visits = list(route.visits())
                routes_data.append(route_visits)
            
            migration_data = {
                'island_id': self.island_id,
                'cost': current_cost,
                'solution': {
                    'routes': routes_data,
                    'num_vehicles': len(self._best.routes())
                },
                'type': 'best',  # 标记为最优解
                'timestamp': time.time()
            }
            
            try:
                self.migration_queue.put(migration_data, timeout=0.001)
                self.last_sent_cost = current_cost
                self.total_migrations_sent += 1
            except queue.Full:
                pass  # 队列满了就跳过
                
        except Exception as e:
            print(f"[岛屿 {self.island_id}] 发送最优解时出错: {e}")
    
    def _send_diversity_solutions(self):
        """发送多样性解以增加种群多样性"""
        try:
            # 选择一些不同的可行解
            diversity_solutions = []
            for i in range(min(2, self._pop.num_feasible() - 1)):
                sol = self._pop._feas[i + 1].solution  # 跳过最优解
                if sol.is_feasible():
                    routes_data = []
                    for route in sol.routes():
                        route_visits = list(route.visits())
                        routes_data.append(route_visits)
                    
                    diversity_data = {
                        'island_id': self.island_id,
                        'cost': sol.distance(),
                        'solution': {
                            'routes': routes_data,
                            'num_vehicles': len(sol.routes())
                        },
                        'type': 'diversity',  # 标记为多样性解
                        'timestamp': time.time()
                    }
                    
                    try:
                        self.migration_queue.put(diversity_data, timeout=0.001)
                        self.total_migrations_sent += 1
                    except queue.Full:
                        break
                        
        except Exception as e:
            print(f"[岛屿 {self.island_id}] 发送多样性解时出错: {e}")
    
    def run(self, stop, collect_stats=True, display=False, display_interval=5.0):
        """重写run方法以支持迁移机制"""
        try:
            print_progress = ProgressPrinter(display, display_interval)
        except TypeError:
            print_progress = ProgressPrinter(display)
        
        print_progress.start(self._data)

        start = time.perf_counter()
        stats = Statistics(collect_stats=collect_stats)
        iters = 0
        iters_no_improvement = 1

        for sol in self._initial_solutions:
            self._pop.add(sol, self._cost_evaluator)

        # 为自定义停止条件设置种群引用
        if hasattr(stop, 'set_population_reference'):
            stop.set_population_reference(self._pop, self._cost_evaluator)

        restart_threshold = 20000
        try:
            restart_threshold = getattr(self._params, "num_iters_no_improvement", restart_threshold)
        except (AttributeError, TypeError):
            pass
            
        while not stop(self._cost_evaluator.cost(self._best)):
            iters += 1
            self.migration_counter += 1

            if iters_no_improvement == restart_threshold:
                print_progress.restart()
                iters_no_improvement = 1
                self._pop.clear()
                for sol in self._initial_solutions:
                    self._pop.add(sol, self._cost_evaluator)

            curr_best = self._cost_evaluator.cost(self._best)

            # 动态迁移策略
            self._update_stagnation_status(curr_best)
            self._adaptive_migration_control()
            
            # 处理迁移（智能频率控制）
            if self._should_migrate():
                self.process_migrations()
                self.send_migration()

            # 使用混合选择策略选择父代
            parents = self._selector.select(self._pop, self._rng, self._cost_evaluator)
            offspring = self._crossover(
                parents, self._data, self._cost_evaluator, self._rng
            )
            
            # 改进子代
            self._improve_offspring(offspring)
            
            # 对父代进行奖励更新
            for parent in parents:
                self._selector.update_reward(parent, offspring, self._cost_evaluator)

            new_best = self._cost_evaluator.cost(self._best)

            if new_best < curr_best:
                iters_no_improvement = 1
            else:
                iters_no_improvement += 1

            stats.collect_from(self._pop, self._cost_evaluator)
            print_progress.iteration(stats)

        end = time.perf_counter() - start
        res = Result(self._best, stats, iters, end)
        print_progress.end(res)
        
        # 输出迁移统计信息
        if self.island_id is not None:
            success_rate = (self.successful_migrations / max(self.total_migrations_sent, 1)) * 100
            print(f"[岛屿 {self.island_id}] 迁移统计: 发送 {self.total_migrations_sent}, 成功接受 {self.successful_migrations}, 成功率 {success_rate:.1f}%")
        
        return res
    
    def _update_stagnation_status(self, current_cost):
        """更新停滞状态"""
        if current_cost < self.last_best_cost:
            self.stagnation_counter = 0
            self.last_best_cost = current_cost
        else:
            self.stagnation_counter += 1
    
    def _adaptive_migration_control(self):
        """自适应迁移频率控制"""
        # 根据停滞情况动态调整迁移间隔
        if self.stagnation_counter > 500:
            # 严重停滞，增加迁移频率
            self.migration_interval = max(20, self.base_migration_interval // 4)
        elif self.stagnation_counter > 200:
            # 中度停滞，适度增加迁移频率
            self.migration_interval = max(50, self.base_migration_interval // 2)
        else:
            # 正常状态，使用基础间隔
            self.migration_interval = self.base_migration_interval
    
    def _should_migrate(self):
        """智能决定是否进行迁移 - 增加隔离期检查"""
        # 隔离期检查 - 在隔离期内不进行迁移
        if self.migration_counter < self.min_isolation_period:
            return False
        
        # 基础间隔检查
        if self.migration_counter % self.migration_interval != 0:
            return False
        
        # 增加随机性，避免所有岛屿同时迁移
        import random
        base_probability = 0.3  # 降低基础概率从0.7到0.3
        
        # 长期停滞时才增加迁移概率（阈值提高）
        if self.stagnation_counter > 1000:  # 从200提高到1000
            base_probability = min(0.6, base_probability + self.stagnation_counter / 2000)  # 降低最大概率
        
        return random.random() < base_probability


class IslandWorker:
    """岛屿模型中的单个工作进程"""
    
    @staticmethod
    def run_island(island_id, data_dict, seed_base, hybrid_params, custom_neighbourhood_params, 
                   custom_swap_star_params, stop_time, migration_queue, result_queue, 
                   migration_interval=50):
 
        try:
            # 反序列化数据并重建PyVRP数据结构
            data = IslandWorker._rebuild_data_from_dict(data_dict)
            
            # 为每个岛屿使用不同的随机种子
            island_seed = seed_base + island_id * 1000
            
            # 创建岛屿的遗传算法实例（使用支持迁移的版本）
            algo, rng = create_island_genetic_algorithm(
                data=data,
                seed=island_seed,
                hybrid_params=hybrid_params,
                custom_neighbourhood_params=custom_neighbourhood_params,
                custom_swap_star_params=custom_swap_star_params,
                island_id=island_id,
                migration_queue=migration_queue
            )
            
            algo.migration_interval = migration_interval
            
            print(f"[岛屿 {island_id}] 开始进化，种子: {island_seed}")
            
            # 运行岛屿进化
            from pyvrp.stop import MaxRuntime
            stop_criteria = MaxRuntime(stop_time)
            
            res = algo.run(
                stop=stop_criteria,
                collect_stats=False,
                display=False
            )
            
            # 发送最终结果
            if res.best.is_feasible():
                best_cost = res.best.distance()
                final_data = IslandWorker._serialize_solution(res.best)
                result_queue.put((island_id, best_cost, final_data))
                print(f"[岛屿 {island_id}] 完成，最佳成本: {best_cost:.2f}")
            else:
                print(f"[岛屿 {island_id}] 完成，未找到可行解")
                
        except Exception as e:
            print(f"[岛屿 {island_id}] 错误: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def _serialize_solution(solution):
        """序列化解决方案以便在进程间传递"""
        try:
            # 提取解决方案的关键信息
            routes_data = []
            for route in solution.routes():
                route_visits = list(route.visits())
                routes_data.append(route_visits)
            
            solution_data = {
                'routes': routes_data,
                'distance': solution.distance(),
                'is_feasible': solution.is_feasible()
            }
            return solution_data
        except Exception as e:
            print(f"序列化解决方案时出错: {e}")
            return None
    
    @staticmethod
    def _rebuild_data_from_dict(data_dict):
        """从字典重建PyVRP数据结构"""
        try:
            # 这里需要重建Model和data
            # 由于PyVRP的数据结构比较复杂，我们传递构建参数而不是直接序列化data
            from pyvrp import Model
            
            m = Model()
            
            # 重建车辆类型
            vehicle_info = data_dict['vehicle']
            m.add_vehicle_type(
                vehicle_info['num_vehicles'], 
                capacity=vehicle_info['capacity'],
                tw_early=vehicle_info['tw_early'],
                tw_late=vehicle_info['tw_late']
            )
            
            # 重建仓库
            depot_info = data_dict['depot']
            m.add_depot(
                x=depot_info['x'],
                y=depot_info['y'], 
                tw_early=depot_info['tw_early'],
                tw_late=depot_info['tw_late']
            )
            
            # 重建客户
            for client_info in data_dict['clients']:
                m.add_client(
                    x=client_info['x'],
                    y=client_info['y'],
                    tw_early=client_info['tw_early'],
                    tw_late=client_info['tw_late'],
                    delivery=client_info['delivery'],
                    service_duration=client_info['service_duration']
                )
            
            # 重建边
            for edge_info in data_dict['edges']:
                frm = m.locations[edge_info['from_idx']]
                to = m.locations[edge_info['to_idx']]
                m.add_edge(frm, to, distance=edge_info['distance'], duration=edge_info['duration'])
            
            return m.data()
            
        except Exception as e:
            print(f"重建数据时出错: {e}")
            raise
    



class IslandModelSolver:
    """
    岛屿模型并行求解器
    实现多子种群并行进化和个体迁移机制
    """
    
    def __init__(self, num_islands=4, migration_interval=50):
        """
        初始化岛屿模型求解器
        
        参数:
        - num_islands: 岛屿（子种群）数量
        - migration_interval: 迁移间隔（迭代次数）
        """
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.processes = []
        self.migration_queue = None
        self.result_queue = None
        
    def solve(self, data, hybrid_params, custom_neighbourhood_params, 
              custom_swap_star_params, runtime_limit, seed_base=42):
        """
        使用岛屿模型求解VRP问题
        
        参数:
        - data: PyVRP数据对象
        - hybrid_params: 混合算法参数
        - custom_neighbourhood_params: 邻域参数
        - custom_swap_star_params: SwapStar参数
        - runtime_limit: 运行时间限制（秒）
        - seed_base: 基础随机种子
        
        返回:
        - 最优解信息字典
        """
        print(f"启动岛屿模型并行求解，{self.num_islands}个岛屿")
        
        # 创建进程间通信队列
        self.migration_queue = Queue(maxsize=100)  # 迁移队列
        self.result_queue = Queue(maxsize=50)      # 结果队列
        
        try:
            # 序列化数据以传递给子进程
            data_dict = self._serialize_data(data)
            
            # 启动岛屿进程
            self.processes = []
            for island_id in range(self.num_islands):
                p = Process(
                    target=IslandWorker.run_island,
                    args=(
                        island_id, data_dict, seed_base, hybrid_params,
                        custom_neighbourhood_params, custom_swap_star_params,
                        runtime_limit, self.migration_queue, self.result_queue,
                        self.migration_interval
                    )
                )
                p.start()
                self.processes.append(p)
                print(f"启动岛屿 {island_id}")
            
            # 收集结果
            best_cost = float('inf')
            best_solution_data = None
            best_island = -1
            
                   # 等待所有进程完成
            for p in self.processes:
                p.join()

            
            # 收集所有结果
            results = []
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get(timeout=1.0)
                    results.append(result)
                except queue.Empty:
                    break
            
            # 找到最优解
            for island_id, cost, solution_data in results:
                if cost < best_cost:
                    best_cost = cost
                    best_solution_data = solution_data
                    best_island = island_id
            
            if best_solution_data is not None:
                print(f"岛屿模型求解完成！最优解来自岛屿 {best_island}，成本: {best_cost:.2f}")
                return {
                    'cost': best_cost,
                    'solution_data': best_solution_data,
                    'best_island': best_island,
                    'all_results': results
                }
            else:
                print("岛屿模型求解完成，但未找到可行解")
                return None
                
        except Exception as e:
            print(f"岛屿模型求解时出错: {e}")
            # 清理进程
            self._cleanup_processes()
            raise
        finally:
            # 确保清理资源
            self._cleanup_processes()
    
    def _serialize_data(self, data):
        """将PyVRP数据序列化为可传递给子进程的字典"""
        try:
            
            raise NotImplementedError("需要在调用层面传递构建参数")
            
        except Exception as e:
            print(f"序列化数据时出错: {e}")
            raise
    
    def _cleanup_processes(self):
        """清理所有子进程"""
        try:
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.0)
                    if p.is_alive():
                        p.kill()
        except Exception as e:
            print(f"清理进程时出错: {e}")


class DiverseSearchConfigGenerator:
    """
    多样化搜索配置生成器
    
    为不同岛屿生成不同的局部搜索参数配置，包括：
    1. 邻域参数（NeighbourhoodParams）多样化
    2. 搜索算子选择多样化  
    3. SwapStar参数多样化
    
    确保不同岛屿使用显著不同的搜索策略，提高全局搜索的多样性。
    """
    
    @staticmethod
    def get_island_search_config(island_id):
        """
        为指定岛屿生成搜索参数配置
        
        参数:
        - island_id: 岛屿编号
        
        返回:
        - 包含邻域参数、算子配置和SwapStar参数的字典
        """
        import hashlib
        import random
        
        # 使用岛屿ID生成确定性的随机种子
        seed = int(hashlib.md5(f"search_{island_id}".encode()).hexdigest()[:8], 16)
        temp_random = random.Random(seed)
        
        # 定义10种邻域搜索策略 - 大幅增加差异化
        neighbourhood_strategies = [
            # 策略0：极密集全覆盖邻域
            {
                'weight_wait_time': 2.0,
                'weight_time_warp': 5.0,
                'nb_granular': 100,
                'symmetric_proximity': True,
                'symmetric_neighbours': True
            },
            # 策略1：密集高权重邻域
            {
                'weight_wait_time': 1.5,
                'weight_time_warp': 4.0,
                'nb_granular': 70,
                'symmetric_proximity': True,
                'symmetric_neighbours': False
            },
            # 策略2：中密集均衡邻域
            {
                'weight_wait_time': 0.8,
                'weight_time_warp': 2.5,
                'nb_granular': 50,
                'symmetric_proximity': True,
                'symmetric_neighbours': False
            },
            # 策略3：中等非对称邻域
            {
                'weight_wait_time': 0.4,
                'weight_time_warp': 1.5,
                'nb_granular': 30,
                'symmetric_proximity': False,
                'symmetric_neighbours': True
            },
            # 策略4：稀疏低权重邻域
            {
                'weight_wait_time': 0.2,
                'weight_time_warp': 0.8,
                'nb_granular': 20,
                'symmetric_proximity': False,
                'symmetric_neighbours': False
            },
            # 策略5：超稀疏时间导向邻域
            {
                'weight_wait_time': 0.05,
                'weight_time_warp': 0.3,
                'nb_granular': 12,
                'symmetric_proximity': False,
                'symmetric_neighbours': True
            },
            # 策略6：极稀疏距离导向邻域
            {
                'weight_wait_time': 0.01,
                'weight_time_warp': 0.1,
                'nb_granular': 8,
                'symmetric_proximity': True,
                'symmetric_neighbours': False
            },
            # 策略7：最小邻域
            {
                'weight_wait_time': 0.005,
                'weight_time_warp': 0.05,
                'nb_granular': 5,
                'symmetric_proximity': False,
                'symmetric_neighbours': False
            },
            # 策略8：高时间扭曲权重邻域
            {
                'weight_wait_time': 0.1,
                'weight_time_warp': 10.0,
                'nb_granular': 40,
                'symmetric_proximity': True,
                'symmetric_neighbours': True
            },
            # 策略9：高等待时间权重邻域
            {
                'weight_wait_time': 3.0,
                'weight_time_warp': 0.5,
                'nb_granular': 35,
                'symmetric_proximity': False,
                'symmetric_neighbours': True
            }
        ]
        
        # 定义8种算子选择策略 - 更激进的差异化
        operator_strategies = [
            # 策略A：极简核心算子
            {
                'focus': 'minimal_core',
                'excluded_node_ops': ['Exchange20', 'Exchange21', 'Exchange22', 'Exchange30', 'Exchange31', 'Exchange32', 'Exchange33', 'SwapTails'],
                'description': '极简核心算子（仅Exchange10/11）'
            },
            # 策略B：单一算子专精
            {
                'focus': 'single_operator',
                'excluded_node_ops': ['Exchange11', 'Exchange20', 'Exchange21', 'Exchange22', 'Exchange30', 'Exchange31', 'Exchange32', 'Exchange33', 'SwapTails'],
                'description': '单一算子专精（仅Exchange10）'
            },
            # 策略C：大规模移动专精
            {
                'focus': 'large_moves_only', 
                'excluded_node_ops': ['Exchange10', 'Exchange11', 'SwapTails', 'TripRelocate'],
                'description': '大规模移动专精'
            },
            # 策略D：中等规模平衡
            {
                'focus': 'medium_balance',
                'excluded_node_ops': ['Exchange30', 'Exchange31', 'Exchange32', 'Exchange33'],
                'description': '中等规模平衡搜索'
            },
            # 策略E：交换算子主导
            {
                'focus': 'exchange_focus',
                'excluded_node_ops': ['SwapTails', 'TripRelocate'],
                'description': '交换算子主导'
            },
            # 策略F：移动算子主导
            {
                'focus': 'relocate_focus',
                'excluded_node_ops': ['Exchange20', 'Exchange21', 'Exchange22', 'Exchange30', 'Exchange31', 'Exchange32'],
                'description': '移动算子主导'
            },
            # 策略G：全算子策略
            {
                'focus': 'full_coverage',
                'excluded_node_ops': [],
                'description': '全算子覆盖搜索'
            },
            # 策略H：高复杂度算子
            {
                'focus': 'complex_ops',
                'excluded_node_ops': ['Exchange10', 'Exchange11', 'Exchange20', 'Exchange21'],
                'description': '高复杂度算子专精'
            }
        ]
        
        # SwapStar参数多样化（10种配置）- 在有效范围内[0,1]
        swap_star_params = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
        
        # 搜索强度多样化（新增）
        search_intensity_configs = [
            {'name': 'light', 'factor': 0.5},      # 轻度搜索
            {'name': 'normal', 'factor': 1.0},     # 标准搜索  
            {'name': 'intensive', 'factor': 2.0},  # 密集搜索
            {'name': 'extreme', 'factor': 3.0},    # 极限搜索
        ]
        
        # 基于岛屿ID选择策略（确保可重复性）
        neighbourhood_idx = island_id % len(neighbourhood_strategies)
        operator_idx = island_id % len(operator_strategies)
        swap_star_idx = island_id % len(swap_star_params)
        intensity_idx = island_id % len(search_intensity_configs)
        
        # 添加一些随机微调以增加多样性
        neighbourhood_config = neighbourhood_strategies[neighbourhood_idx].copy()
        
        # 大幅微调邻域参数以增加差异
        neighbourhood_config['weight_wait_time'] *= temp_random.uniform(0.3, 2.0)  # 更大变化范围
        neighbourhood_config['weight_time_warp'] *= temp_random.uniform(0.2, 3.0)  # 更大变化范围
        neighbourhood_config['nb_granular'] = max(5, int(neighbourhood_config['nb_granular'] * temp_random.uniform(0.5, 1.8)))  # 更大变化范围
        
        # 扩大边界检查范围
        neighbourhood_config['weight_wait_time'] = max(0.001, min(5.0, neighbourhood_config['weight_wait_time']))
        neighbourhood_config['weight_time_warp'] = max(0.01, min(15.0, neighbourhood_config['weight_time_warp']))
        neighbourhood_config['nb_granular'] = max(3, min(150, neighbourhood_config['nb_granular']))
        
        return {
            'neighbourhood_params': NeighbourhoodParams(**neighbourhood_config),
            'operator_config': operator_strategies[operator_idx],
            'swap_star_tolerance': swap_star_params[swap_star_idx],
            'search_intensity': search_intensity_configs[intensity_idx],
            'strategy_description': f"邻域{neighbourhood_idx}-算子{operator_idx}-SwapStar{swap_star_idx}-强度{intensity_idx}"
        }
    
    @staticmethod
    def create_diverse_local_search(data, rng, island_id):
        """
        为指定岛屿创建多样化的局部搜索实例
        
        参数:
        - data: PyVRP问题数据
        - rng: 随机数生成器
        - island_id: 岛屿ID
        
        返回:
        - 配置好的LocalSearch实例
        """
        # 获取岛屿特定的搜索配置
        search_config = DiverseSearchConfigGenerator.get_island_search_config(island_id)
        
        # 使用岛屿特定的邻域参数计算邻居
        neighbours = compute_neighbours(data, search_config['neighbourhood_params'])
        ls = LocalSearch(data, rng, neighbours)
        
        # 根据策略添加节点算子
        operator_config = search_config['operator_config']
        excluded_ops = set(operator_config['excluded_node_ops'])
        
        for node_op in NODE_OPERATORS:
            if node_op.__name__ not in excluded_ops:
                try:
                    if hasattr(node_op, 'supports') and node_op.supports(data):
                        ls.add_node_operator(node_op(data))
                    else:
                        ls.add_node_operator(node_op(data))
                except Exception as e:
                    print(f"[岛屿 {island_id}] 添加节点算子 {node_op.__name__} 时出错: {e}")
        
        # 添加路径算子，SwapStar使用岛屿特定参数
        for route_op in ROUTE_OPERATORS:
            try:
                if route_op.__name__ == 'SwapStar':
                    # 使用岛屿特定的overlap_tolerance参数
                    swap_star = SwapStar(data, overlap_tolerance=search_config['swap_star_tolerance'])
                    ls.add_route_operator(swap_star)
                else:
                    if hasattr(route_op, 'supports') and route_op.supports(data):
                        ls.add_route_operator(route_op(data))
                    else:
                        ls.add_route_operator(route_op(data))
            except Exception as e:
                print(f"[岛屿 {island_id}] 添加路径算子 {route_op.__name__} 时出错: {e}")
        
        # 暂时禁用强度多样化包装器以解决收敛问题
        # intensity_config = search_config['search_intensity']
        # wrapped_ls = DiverseIntensityLocalSearch(ls, intensity_config)
        
        # 输出配置信息（可选，用于调试）
        if island_id is not None and island_id < 10:  # 只输出前10个岛屿的配置信息
            print(f"[岛屿 {island_id}] 搜索配置: {search_config['strategy_description']}")
            print(f"  - 邻居数: {search_config['neighbourhood_params'].nb_granular}")
            print(f"  - 算子策略: {operator_config['description']}")
            print(f"  - SwapStar参数: {search_config['swap_star_tolerance']}")
            # print(f"  - 搜索强度: {intensity_config['name']} (因子: {intensity_config['factor']})")
            print(f"  - 搜索强度: 标准 (暂时禁用强度多样化)")
        
        return ls  # 直接返回标准LocalSearch，不使用强度包装器


class DiverseInitialSolutionGenerator:
    """
    多样化初始解生成器
    
    提供多种不同的初始解生成策略，确保不同岛屿使用不同的初始化方法，
    从而增加种群的多样性，提高全局搜索能力。
    """
    
    @staticmethod
    def generate_random_solution(data, rng):
        """标准随机解生成（PyVRP默认方法）"""
        return Solution.make_random(data, rng)
    
    @staticmethod
    def generate_nearest_neighbor_solution(data, rng):
        """最近邻算法生成初始解"""
        try:
            # 创建一个基于最近邻启发式的解
            clients = list(range(data.num_depots, data.num_locations))
            rng_copy = RandomNumberGenerator(rng())  # 创建独立的随机数生成器
            
            routes = []
            unvisited = set(clients)
            
            # 为每个可用车辆尝试构建路径
            vehicle_count = 0
            while unvisited and vehicle_count < data.num_vehicles:
                route = []
                current_location = 0  # 从仓库开始
                
                # 构建一条路径
                while unvisited:
                    # 找到最近的未访问客户
                    nearest_client = None
                    min_distance = float('inf')
                    
                    for client in unvisited:
                        try:
                            # 计算距离（使用data中的距离矩阵）
                            distance = data.distance(current_location, client)
                            if distance < min_distance:
                                min_distance = distance
                                nearest_client = client
                        except:
                            # 如果无法获取距离，使用欧几里得距离
                            loc1 = data.location(current_location)
                            loc2 = data.location(client)
                            distance = ((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2) ** 0.5
                            if distance < min_distance:
                                min_distance = distance
                                nearest_client = client
                    
                    if nearest_client is None:
                        break
                    
                    # 检查容量约束（简化检查）
                    if len(route) >= max(1, len(clients) // data.num_vehicles):
                        break
                    
                    route.append(nearest_client)
                    unvisited.remove(nearest_client)
                    current_location = nearest_client
                
                if route:
                    routes.append(route)
                    vehicle_count += 1
                else:
                    break
            
            # 将剩余客户随机分配到现有路径
            remaining_clients = list(unvisited)
            if remaining_clients and routes:
                for client in remaining_clients:
                    route_idx = rng.randint(len(routes))
                    routes[route_idx].append(client)
            
            # 如果没有路径，回退到随机方法
            if not routes:
                return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
            
            return Solution(data, routes)
            
        except Exception as e:
            # 如果最近邻方法失败，回退到随机方法
            return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
    
    @staticmethod
    def generate_sweep_solution(data, rng):
        """扫描算法生成初始解"""
        try:
            clients = list(range(data.num_depots, data.num_locations))
            if not clients:
                return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
            
            # 计算每个客户相对于仓库的角度
            depot = data.location(0)
            client_angles = []
            
            for client_idx in clients:
                client = data.location(client_idx)
                angle = math.atan2(client.y - depot.y, client.x - depot.x)
                client_angles.append((angle, client_idx))
            
            # 按角度排序
            client_angles.sort()
            
            # 随机选择起始角度
            start_idx = rng.randint(len(client_angles))
            sorted_clients = [client_angles[(start_idx + i) % len(client_angles)][1] 
                            for i in range(len(client_angles))]
            
            # 按扫描顺序分配到车辆
            routes = []
            clients_per_vehicle = max(1, len(sorted_clients) // data.num_vehicles)
            
            for i in range(0, len(sorted_clients), clients_per_vehicle):
                route = sorted_clients[i:i + clients_per_vehicle]
                if route:
                    routes.append(route)
            
            # 限制路径数量不超过可用车辆数
            while len(routes) > data.num_vehicles and len(routes) > 1:
                # 合并最短的两条路径
                min_idx = min(range(len(routes)), key=lambda i: len(routes[i]))
                route_to_merge = routes.pop(min_idx)
                if routes:
                    routes[0].extend(route_to_merge)
            
            return Solution(data, routes) if routes else DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
            
        except Exception as e:
            return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
    
    @staticmethod
    def generate_clustered_solution(data, rng):
        """基于聚类的初始解生成"""
        try:
            clients = list(range(data.num_depots, data.num_locations))
            if len(clients) <= data.num_vehicles:
                # 客户数量少于车辆数，直接分配
                routes = [[client] for client in clients]
                return Solution(data, routes)
            
            # 简单的k-means聚类（简化版）
            num_clusters = min(data.num_vehicles, len(clients))
            
            # 随机选择初始聚类中心
            cluster_centers = rng.sample(clients, num_clusters) if hasattr(rng, 'sample') else \
                             [clients[rng.randint(len(clients))] for _ in range(num_clusters)]
            
            # 分配客户到最近的聚类中心
            clusters = [[] for _ in range(num_clusters)]
            
            for client in clients:
                client_loc = data.location(client)
                min_distance = float('inf')
                best_cluster = 0
                
                for i, center_client in enumerate(cluster_centers):
                    center_loc = data.location(center_client)
                    distance = ((client_loc.x - center_loc.x) ** 2 + 
                              (client_loc.y - center_loc.y) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_cluster = i
                
                clusters[best_cluster].append(client)
            
            # 过滤空聚类
            routes = [cluster for cluster in clusters if cluster]
            
            return Solution(data, routes) if routes else DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
            
        except Exception as e:
            return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
    
    @staticmethod
    def generate_greedy_solution(data, rng):
        """贪心算法生成初始解（优先选择高需求客户）"""
        try:
            clients = list(range(data.num_depots, data.num_locations))
            if not clients:
                return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
            
            # 计算每个客户的"重要性"（基于需求和距离）
            client_priority = []
            depot = data.location(0)
            
            for client_idx in clients:
                client = data.location(client_idx)
                # 计算距离仓库的距离
                distance = ((client.x - depot.x) ** 2 + (client.y - depot.y) ** 2) ** 0.5
                
                # 获取需求量（如果有）
                demand = sum(client.delivery) if hasattr(client, 'delivery') and client.delivery else 1
                
                # 优先级 = 需求 / (距离 + 1)，距离近且需求大的客户优先级高
                priority = demand / (distance + 1)
                client_priority.append((priority, client_idx))
            
            # 按优先级排序（高优先级在前）
            client_priority.sort(reverse=True)
            sorted_clients = [client_idx for _, client_idx in client_priority]
            
            # 按优先级分配到车辆
            routes = []
            clients_per_vehicle = max(1, len(sorted_clients) // data.num_vehicles)
            
            for i in range(0, len(sorted_clients), clients_per_vehicle):
                route = sorted_clients[i:i + clients_per_vehicle]
                if route:
                    routes.append(route)
            
            # 限制路径数量
            while len(routes) > data.num_vehicles and len(routes) > 1:
                shortest_route = min(routes, key=len)
                routes.remove(shortest_route)
                if routes:
                    routes[0].extend(shortest_route)
            
            return Solution(data, routes) if routes else DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
            
        except Exception as e:
            return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
    
    @staticmethod
    def generate_time_window_aware_solution(data, rng):
        """考虑时间窗的初始解生成"""
        try:
            clients = list(range(data.num_depots, data.num_locations))
            if not clients:
                return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
            
            # 按时间窗早期开始时间排序
            client_tw = []
            for client_idx in clients:
                client = data.location(client_idx)
                tw_early = getattr(client, 'tw_early', 0)
                client_tw.append((tw_early, client_idx))
            
            # 排序并添加随机性
            client_tw.sort()
            
            # 添加一些随机扰动以增加多样性
            if len(client_tw) > 2:
                # 随机交换一些相邻的客户
                for _ in range(len(client_tw) // 4):
                    idx = rng.randint(len(client_tw) - 1)
                    client_tw[idx], client_tw[idx + 1] = client_tw[idx + 1], client_tw[idx]
            
            sorted_clients = [client_idx for _, client_idx in client_tw]
            
            # 分配到车辆
            routes = []
            clients_per_vehicle = max(1, len(sorted_clients) // data.num_vehicles)
            
            for i in range(0, len(sorted_clients), clients_per_vehicle):
                route = sorted_clients[i:i + clients_per_vehicle]
                if route:
                    routes.append(route)
            
            # 限制路径数量
            while len(routes) > data.num_vehicles and len(routes) > 1:
                routes[0].extend(routes.pop())
            
            return Solution(data, routes) if routes else DiverseInitialSolutionGenerator.generate_random_solution(data, rng)
            
        except Exception as e:
            return DiverseInitialSolutionGenerator.generate_random_solution(data, rng)

    @staticmethod
    def generate_diverse_initial_solutions(data, rng, population_size, island_id=None):
        """
        为指定的岛屿生成多样化的初始解集合
        
        参数:
        - data: PyVRP问题数据
        - rng: 随机数生成器
        - population_size: 需要生成的解的数量
        - island_id: 岛屿ID，用于确定生成策略的分布
        
        返回:
        - 初始解列表
        """
        solutions = []
        
        # 定义不同的生成方法
        generation_methods = [
            DiverseInitialSolutionGenerator.generate_random_solution,
            DiverseInitialSolutionGenerator.generate_nearest_neighbor_solution,
            DiverseInitialSolutionGenerator.generate_sweep_solution,
            DiverseInitialSolutionGenerator.generate_clustered_solution,
            DiverseInitialSolutionGenerator.generate_greedy_solution,
            DiverseInitialSolutionGenerator.generate_time_window_aware_solution,
        ]
        
        # 基于岛屿ID确定方法分布（如果提供了island_id）
        if island_id is not None:
            # 不同岛屿偏好不同的初始化方法
            method_weights = DiverseInitialSolutionGenerator._get_method_weights_for_island(island_id)
        else:
            # 均匀分布
            method_weights = [1.0] * len(generation_methods)
        
        # 生成解
        for i in range(population_size):
            # 根据权重选择生成方法
            method_idx = DiverseInitialSolutionGenerator._weighted_choice(method_weights, rng)
            method = generation_methods[method_idx]
            
            try:
                solution = method(data, rng)
                solutions.append(solution)
            except Exception as e:
                # 如果特定方法失败，使用随机方法作为后备
                solutions.append(DiverseInitialSolutionGenerator.generate_random_solution(data, rng))
        
        return solutions
    
    @staticmethod
    def _get_method_weights_for_island(island_id):
        """为不同岛屿返回不同的方法权重分布"""
        # 6种方法的权重：[随机, 最近邻, 扫描, 聚类, 贪心, 时间窗]
        weight_patterns = [
            [0.7, 0.1, 0.1, 0.05, 0.03, 0.02],  # 主要随机
            [0.3, 0.4, 0.1, 0.1, 0.05, 0.05],   # 最近邻主导
            [0.2, 0.1, 0.5, 0.1, 0.05, 0.05],   # 扫描主导
            [0.2, 0.1, 0.1, 0.4, 0.1, 0.1],     # 聚类主导
            [0.2, 0.1, 0.1, 0.1, 0.4, 0.1],     # 贪心主导
            [0.2, 0.1, 0.1, 0.1, 0.1, 0.4],     # 时间窗主导
            [0.3, 0.2, 0.2, 0.1, 0.1, 0.1],     # 几何方法主导
            [0.15, 0.15, 0.15, 0.15, 0.2, 0.2], # 启发式主导
            [0.4, 0.15, 0.15, 0.1, 0.1, 0.1],   # 随机和几何混合
            [0.1, 0.3, 0.3, 0.1, 0.1, 0.1],     # 空间方法主导
        ]
        
        pattern_idx = island_id % len(weight_patterns)
        return weight_patterns[pattern_idx]
    
    @staticmethod
    def _weighted_choice(weights, rng):
        """根据权重进行随机选择"""
        total = sum(weights)
        r = rng.rand() * total
        
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return i
        
        return len(weights) - 1  # 后备选择


def create_island_genetic_algorithm(data, 
    seed=42, 
    hybrid_params=None, 
    custom_neighbourhood_params=None,
    custom_swap_star_params=0.05,
    island_id=None,
    migration_queue=None,
    is_distance_optimization=False
):
    """创建支持迁移的岛屿遗传算法"""
    from pyvrp.solve import SolveParams
    from pyvrp.GeneticAlgorithm import GeneticAlgorithmParams
    from pyvrp.PenaltyManager import PenaltyManager
    from pyvrp.Population import Population
    from pyvrp._pyvrp import RandomNumberGenerator, Solution
    from pyvrp.diversity import broken_pairs_distance as bpd
    from pyvrp.crossover import selective_route_exchange as srex
    from pyvrp.crossover import ordered_crossover as ox
    from pyvrp.search import SwapStar
    from pyvrp.search import NeighbourhoodParams
    
    genetic_params = GeneticAlgorithmParams()
    
    # 解算器参数
    params = SolveParams(
        genetic=genetic_params,
        neighbourhood=custom_neighbourhood_params or NeighbourhoodParams(),
    )
    
    # 初始化解算组件
    rng = RandomNumberGenerator(seed=seed)
    from pyvrp.search import compute_neighbours, LocalSearch, NODE_OPERATORS, ROUTE_OPERATORS
    
    # 使用多样化的局部搜索（如果指定了岛屿ID）
    if island_id is not None:
        ls = DiverseSearchConfigGenerator.create_diverse_local_search(data, rng, island_id)
    else:
        # 默认使用标准配置
        neighbours = compute_neighbours(data, params.neighbourhood)
        ls = LocalSearch(data, rng, neighbours)
        
        # 添加搜索算子
        for node_op in NODE_OPERATORS:
            try:
                if hasattr(node_op, 'supports') and node_op.supports(data):
                    ls.add_node_operator(node_op(data))
                else:
                    ls.add_node_operator(node_op(data))
            except Exception as e:
                print(f"添加节点算子 {node_op.__name__} 时出错: {e}")
        
        # 特别处理ROUTE_OPERATORS
        for route_op in ROUTE_OPERATORS:
            try:
                if route_op.__name__ == 'SwapStar':
                    swap_star_instance = SwapStar(data, overlap_tolerance=custom_swap_star_params)
                    ls.add_route_operator(swap_star_instance)
                else:
                    if hasattr(route_op, 'supports') and route_op.supports(data):
                        ls.add_route_operator(route_op(data))
                    else:
                        ls.add_route_operator(route_op(data))
            except Exception as e:
                print(f"添加路径算子 {route_op.__name__} 时出错: {e}")
    
    pm = PenaltyManager.init_from(data, params.penalty)
    pop = Population(bpd, params.population)
    
    # 使用多样化的初始解生成策略
    init = DiverseInitialSolutionGenerator.generate_diverse_initial_solutions(
        data, rng, params.population.min_pop_size, island_id
    )
    
    crossover = srex if data.num_vehicles > 1 else ox
    
    # 初始化岛屿遗传算法
    algo = IslandGeneticAlgorithm(
        data, pm, rng, pop, ls, crossover, init, 
        params=genetic_params, 
        hybrid_params=hybrid_params,
        island_id=island_id,
        migration_queue=migration_queue,
        is_distance_optimization=is_distance_optimization
    )
    
    return algo, rng


def create_island_model_solver(num_islands=4, migration_interval=50):
    """创建岛屿模型求解器的工厂函数"""
    return IslandModelSolver(num_islands=num_islands, migration_interval=migration_interval)
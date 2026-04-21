#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python 装饰器 (Decorator) 演示示例

本文件演示了Python装饰器的几种常见用法，包括：
1. 基础装饰器
2. 带参数的装饰器
3. 计时装饰器
4. 使用functools.wraps保留原函数属性
5. 装饰器叠加使用
"""

import time
import functools
from typing import Any, Callable


# ==================== 1. 基础装饰器 ====================
def simple_decorator(func: Callable) -> Callable:
    """
    基础装饰器示例：在函数执行前后打印信息
    
    Args:
        func: 被装饰的函数
        
    Returns:
        包装后的函数
    """
    def wrapper() -> Any:
        print("✨ 函数执行前的额外操作")
        result = func()
        print("✨ 函数执行后的额外操作")
        return result
    return wrapper


@simple_decorator
def say_hello() -> None:
    """被装饰的简单函数"""
    print("Hello, World!")


# ==================== 2. 带参数的装饰器 ====================
def log_decorator(level: str = "INFO") -> Callable:
    """
    带参数的装饰器：根据日志级别记录函数执行
    
    Args:
        level: 日志级别（DEBUG, INFO, WARNING, ERROR）
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            print(f"[{level}] 函数 {func.__name__} 开始执行")
            result = func(*args, **kwargs)
            print(f"[{level}] 函数 {func.__name__} 执行完成")
            return result
        return wrapper
    return decorator


@log_decorator(level="DEBUG")
def multiply(a: int, b: int) -> int:
    """乘法函数"""
    return a * b


# ==================== 3. 计时装饰器 ====================
def timer_decorator(func: Callable) -> Callable:
    """
    计时装饰器：测量函数执行时间
    
    Args:
        func: 被装饰的函数
        
    Returns:
        包装后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"⏱️ 函数 {func.__name__} 执行耗时: {execution_time:.6f} 秒")
        return result
    return wrapper


@timer_decorator
def slow_function() -> int:
    """模拟耗时操作"""
    time.sleep(0.5)  # 模拟耗时操作
    return 42


# ==================== 4. 装饰器叠加使用 ====================
# 装饰器叠加顺序：从下到上执行
@log_decorator(level="INFO")
@timer_decorator
def factorial(n: int) -> int:
    """计算阶乘"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)


# ==================== 5. 类装饰器 ====================
class CountCalls:
    """
    类装饰器：统计函数调用次数
    """
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.call_count = 0
        functools.update_wrapper(self, func)
    
    def __call__(self, *args, **kwargs) -> Any:
        self.call_count += 1
        print(f"函数 {self.func.__name__} 已被调用 {self.call_count} 次")
        return self.func(*args, **kwargs)


@CountCalls
def greet(name: str) -> str:
    """打招呼函数"""
    return f"Hello, {name}!"


# ==================== 主函数 ====================
def main() -> None:
    """主函数：运行所有装饰器示例"""
    print("=" * 50)
    print("Python 装饰器演示")
    print("=" * 50)
    
    # 1. 基础装饰器示例
    print("\n1. 基础装饰器示例:")
    say_hello()
    
    # 2. 带参数的装饰器示例
    print("\n2. 带参数的装饰器示例:")
    result = multiply(5, 8)
    print(f"乘法结果: {result}")
    
    # 3. 计时装饰器示例
    print("\n3. 计时装饰器示例:")
    answer = slow_function()
    print(f"函数返回值: {answer}")
    
    # 4. 装饰器叠加示例
    print("\n4. 装饰器叠加示例:")
    # 注意：递归函数会导致装饰器被多次调用
    fact_result = factorial(5)
    print(f"5的阶乘: {fact_result}")
    
    # 5. 类装饰器示例
    print("\n5. 类装饰器示例:")
    print(greet("Alice"))
    print(greet("Bob"))
    print(greet("Charlie"))
    
    # 6. 显示函数属性
    print("\n6. 函数属性检查:")
    print(f"multiply函数名: {multiply.__name__}")
    print(f"multiply文档: {multiply.__doc__}")
    
    print("\n" + "=" * 50)
    print("演示完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()

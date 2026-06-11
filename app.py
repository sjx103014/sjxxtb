import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
from folium import plugins
import random
import time
import math
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import copy
import heapq
import numpy as np

# ==================== 页面配置 ====================
st.set_page_config(page_title="无人机地面站系统 - 平行偏移绕行", layout="wide")

# ==================== 坐标 ====================
SCHOOL_CENTER_GCJ = [118.7490, 32.2340]
DEFAULT_A_GCJ = [118.746956, 32.232945]
DEFAULT_B_GCJ = [118.751589, 32.235204]

# 高德瓦片地址
GAODE_SATELLITE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
GAODE_VECTOR_URL = "https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
GAODE_SATELLITE_URL_ALT = "https://webst01.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}"

# ==================== 坐标系转换 ====================
def gcj02_to_wgs84(lng, lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat

def wgs84_to_gcj02(lng, lat):
    a = 6378245.0
    ee = 0.00669342162296594323
    if out_of_china(lng, lat):
        return lng, lat
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return mglng, mglat

def transform_lat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transform_lng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
    ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

def out_of_china(lng, lat):
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)

# ==================== 几何辅助函数 ====================
def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def segments_intersect(p1, p2, p3, p4):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def line_intersects_polygon(p1, p2, polygon):
    if point_in_polygon(p1, polygon) or point_in_polygon(p2, polygon):
        return True
    n = len(polygon)
    for i in range(n):
        p3 = polygon[i]
        p4 = polygon[(i + 1) % n]
        if segments_intersect(p1, p2, p3, p4):
            if not (p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4):
                return True
    return False

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# ===== 航点距离抽稀函数 =====
def simplify_path_by_distance(points, min_dist_deg=0.0003):
    """按经纬度距离抽稀航线，小于阈值的航点剔除，保留首尾和拐点"""
    if len(points) <= 2:
        return points
    new_path = [points[0]]
    last = points[0]
    for p in points[1:]:
        if distance(last, p) >= min_dist_deg:
            new_path.append(p)
            last = p
    if new_path[-1] != points[-1]:
        new_path.append(points[-1])
    return new_path

# ==================== 三次B样条路径平滑 ====================
def catmull_rom_spline(points, num_points=6):
    """Catmull-Rom 样条曲线，生成平滑+精简路径"""
    if len(points) < 2:
        return points
    if len(points) == 2:
        return [points[0], points[1]]
    
    extended = [points[0]] + points + [points[-1]]
    spline_points = []
    
    for i in range(len(extended)-3):
        p0, p1, p2, p3 = extended[i], extended[i+1], extended[i+2], extended[i+3]
        for t in np.linspace(0, 1, num_points):
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2 * p1[0]) +
                (-p0[0] + p2[0]) * t +
                (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 +
                (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1]) +
                (-p0[1] + p2[1]) * t +
                (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 +
                (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3
            )
            spline_points.append([x, y])
    unique_points = []
    seen = set()
    for p in spline_points:
        key = (round(p[0], 8), round(p[1], 8))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    full_spline = [points[0]] + unique_points + [points[-1]]
    return simplify_path_by_distance(full_spline, min_dist_deg=0.0003)

# ==================== 障碍物高度与阻挡判断 ====================
def is_obstacle_blocking(obs, flight_height):
    obs_height = obs.get('height', 20)
    return flight_height < obs_height

def is_path_blocked(p1, p2, obstacles_gcj, flight_height):
    for obs in obstacles_gcj:
        if is_obstacle_blocking(obs, flight_height):
            coords = obs.get('polygon', [])
            if coords and len(coords) >= 3:
                if line_intersects_polygon(p1, p2, coords):
                    return True
    return False

# ==================== 单侧绕行路径生成（简化版） ====================
def generate_side_bypass_path(start, end, obstacles_gcj, flight_height, safe_radius, side='left'):
    """
    生成左绕行或右绕行路径
    side: 'left' 或 'right'
    原理：在起点和终点之间添加一个偏移控制点，让路径从障碍物左侧或右侧绕过
    """
    # 获取需要绕行的障碍物
    block_obs = [obs for obs in obstacles_gcj if is_obstacle_blocking(obs, flight_height)]
    if not block_obs:
        return None
    
    # 安全半径转换为度数
    safe_radius_deg = safe_radius / 111000.0
    
    # 计算起点到终点的方向
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-10:
        return None
    
    # 单位方向向量
    ux = dx / length
    uy = dy / length
    
    # 垂直向量（根据左右选择方向）
    if side == 'left':
        # 左绕行：垂直向量向左（逆时针旋转90度）
        perp_x = -uy
        perp_y = ux
    else:
        # 右绕行：垂直向量向右（顺时针旋转90度）
        perp_x = uy
        perp_y = -ux
    
    # 计算所有阻挡障碍物的中心点
    all_centers = []
    for obs in block_obs:
        poly = obs["polygon"]
        if len(poly) >= 3:
            cx = sum(p[0] for p in poly) / len(poly)
            cy = sum(p[1] for p in poly) / len(poly)
            all_centers.append([cx, cy])
    
    if not all_centers:
        return None
    
    # 计算所有障碍物的平均中心
    avg_cx = sum(c[0] for c in all_centers) / len(all_centers)
    avg_cy = sum(c[1] for c in all_centers) / len(all_centers)
    
    # 计算偏移距离（根据安全半径和障碍物大小调整）
    # 获取最远障碍物的距离
    max_dist_to_center = 0
    for obs in block_obs:
        poly = obs["polygon"]
        for p in poly:
            dist = distance([avg_cx, avg_cy], p)
            if dist > max_dist_to_center:
                max_dist_to_center = dist
    
    # 偏移距离 = 障碍物半径 + 安全半径的倍数
    offset_distance = max_dist_to_center + safe_radius_deg * 3
    
    # 生成偏移点
    offset_point = [
        avg_cx + perp_x * offset_distance,
        avg_cy + perp_y * offset_distance
    ]
    
    # 构建路径：起点 -> 偏移点 -> 终点
    path = [start, offset_point, end]
    
    # 碰撞检测
    collision = False
    for i in range(len(path) - 1):
        if is_path_blocked(path[i], path[i+1], obstacles_gcj, flight_height):
            collision = True
            break
    
    if not collision:
        # 平滑路径
        smoothed = catmull_rom_spline(path, num_points=8)
        final_path = simplify_path_by_distance(smoothed)
        return final_path
    
    # 如果失败，尝试更大的偏移距离
    for scale in [4, 5, 6, 7, 8, 10]:
        offset_distance = max_dist_to_center + safe_radius_deg * scale
        offset_point = [
            avg_cx + perp_x * offset_distance,
            avg_cy + perp_y * offset_distance
        ]
        path = [start, offset_point, end]
        
        collision = False
        for i in range(len(path) - 1):
            if is_path_blocked(path[i], path[i+1], obstacles_gcj, flight_height):
                collision = True
                break
        
        if not collision:
            smoothed = catmull_rom_spline(path, num_points=8)
            final_path = simplify_path_by_distance(smoothed)
            return final_path
    
    return None

# ==================== A*路径规划 ====================
def astar_path(start, end, obstacles_gcj, flight_height, safe_radius):
    nodes = [start, end]
    safety = safe_radius / 111000.0 * 2.0

    for obs in obstacles_gcj:
        if not is_obstacle_blocking(obs, flight_height):
            continue
        poly = obs.get('polygon', [])
        if len(poly) < 3:
            continue
        for i, (x, y) in enumerate(poly):
            prev_i = (i - 1) % len(poly)
            prev = poly[prev_i]
            next_i = (i + 1) % len(poly)
            next_p = poly[next_i]

            dx1 = -(y - prev[1])
            dy1 = x - prev[0]
            l1 = math.hypot(dx1, dy1)
            if l1 > 1e-8:
                dx1 /= l1
                dy1 /= l1
            nx1 = x + dx1 * safety
            ny1 = y + dy1 * safety

            dx2 = -(next_p[1] - y)
            dy2 = next_p[0] - x
            l2 = math.hypot(dx2, dy2)
            if l2 > 1e-8:
                dx2 /= l2
                dy2 /= l2
            nx2 = x + dx2 * safety
            ny2 = y + dy2 * safety

            nodes.append([nx1, ny1])
            nodes.append([nx2, ny2])

    unique_nodes = []
    for n in nodes:
        exists = False
        for u in unique_nodes:
            if abs(n[0] - u[0]) < 1e-6 and abs(n[1] - u[1]) < 1e-6:
                exists = True
                break
        if not exists:
            unique_nodes.append(n)

    graph = {i: [] for i in range(len(unique_nodes))}
    for i in range(len(unique_nodes)):
        for j in range(len(unique_nodes)):
            if i == j:
                continue
            if not is_path_blocked(unique_nodes[i], unique_nodes[j], obstacles_gcj, flight_height):
                graph[i].append((j, distance(unique_nodes[i], unique_nodes[j])))

    start_i = -1
    end_i = -1
    for i, n in enumerate(unique_nodes):
        if abs(n[0] - start[0]) < 1e-6 and abs(n[1] - start[1]) < 1e-6:
            start_i = i
        if abs(n[0] - end[0]) < 1e-6 and abs(n[1] - end[1]) < 1e-6:
            end_i = i
    if start_i == -1 or end_i == -1:
        return simplify_path_by_distance([start, end])

    open_heap = []
    heapq.heappush(open_heap, (0, start_i))
    came_from = {}
    g_score = {i: float('inf') for i in range(len(unique_nodes))}
    g_score[start_i] = 0
    f_score = {i: float('inf') for i in range(len(unique_nodes))}
    f_score[start_i] = distance(unique_nodes[start_i], unique_nodes[end_i])

    while open_heap:
        current_f, cur = heapq.heappop(open_heap)
        if cur == end_i:
            path = []
            while cur in came_from:
                path.append(unique_nodes[cur])
                cur = came_from[cur]
            path.append(unique_nodes[start_i])
            path.reverse()
            smooth_path = catmull_rom_spline(path, num_points=5)
            final_path = simplify_path_by_distance(smooth_path)
            return final_path
        for neighbor, w in graph[cur]:
            new_g = g_score[cur] + w
            if new_g < g_score[neighbor]:
                came_from[neighbor] = cur
                g_score[neighbor] = new_g
                f_score[neighbor] = new_g + distance(unique_nodes[neighbor], unique_nodes[end_i])
                heapq.heappush(open_heap, (f_score[neighbor], neighbor))
    return simplify_path_by_distance([start, end])

# ==================== 路径规划主函数 ====================
def create_avoidance_path(start, end, obstacles_gcj, flight_height, safe_radius, strategy):
    # 检查直线是否被阻挡
    straight_blocked = is_path_blocked(start, end, obstacles_gcj, flight_height)
    
    if not straight_blocked:
        path = simplify_path_by_distance([start, end])
        add_gcs_obc_fcu_log(f"航线规划完成 | 类型:直线 | 航点数:{len(path)} | 路径长度:{round(sum([distance(path[i],path[i+1])*111000 for i in range(len(path)-1)]),1)}m")
        return path

    # 直线被阻挡，根据策略选择绕行方式
    if strategy == 'left':
        add_gcs_obc_fcu_log(f"开始航线规划 | 类型:向左绕行 | 飞行高度:{flight_height}m")
        p = generate_side_bypass_path(start, end, obstacles_gcj, flight_height, safe_radius, 'left')
        if p and len(p) >= 2:
            path_length = round(sum([distance(p[i],p[i+1])*111000 for i in range(len(p)-1)]), 1)
            add_gcs_obc_fcu_log(f"航线规划完成 | 类型:向左绕行成功 | 航点数:{len(p)} | 路径长度:{path_length}m")
            return p
        else:
            add_gcs_obc_fcu_log(f"向左绕行失败，降级使用A*算法")
            ast_p = astar_path(start, end, obstacles_gcj, flight_height, safe_radius)
            path_length = round(sum([distance(ast_p[i],ast_p[i+1])*111000 for i in range(len(ast_p)-1)]), 1)
            add_gcs_obc_fcu_log(f"航线规划完成 | 算法:A* (备用) | 航点数:{len(ast_p)} | 路径长度:{path_length}m")
            return ast_p
    elif strategy == 'right':
        add_gcs_obc_fcu_log(f"开始航线规划 | 类型:向右绕行 | 飞行高度:{flight_height}m")
        p = generate_side_bypass_path(start, end, obstacles_gcj, flight_height, safe_radius, 'right')
        if p and len(p) >= 2:
            path_length = round(sum([distance(p[i],p[i+1])*111000 for i in range(len(p)-1)]), 1)
            add_gcs_obc_fcu_log(f"航线规划完成 | 类型:向右绕行成功 | 航点数:{len(p)} | 路径长度:{path_length}m")
            return p
        else:
            add_gcs_obc_fcu_log(f"向右绕行失败，降级使用A*算法")
            ast_p = astar_path(start, end, obstacles_gcj, flight_height, safe_radius)
            path_length = round(sum([distance(ast_p[i],ast_p[i+1])*111000 for i in range(len(ast_p)-1)]), 1)
            add_gcs_obc_fcu_log(f"航线规划完成 | 算法:A* (备用) | 航点数:{len(ast_p)} | 路径长度:{path_length}m")
            return ast_p
    else:
        # 最佳航线 - 使用A*
        add_gcs_obc_fcu_log(f"开始航线规划 | 算法:A* | 障碍物数量:{len([o for o in obstacles_gcj if is_obstacle_blocking(o,flight_height)])}")
        ast_p = astar_path(start, end, obstacles_gcj, flight_height, safe_radius)
        path_length = round(sum([distance(ast_p[i],ast_p[i+1])*111000 for i in range(len(ast_p)-1)]), 1)
        add_gcs_obc_fcu_log(f"航线规划完成 | 算法:A* | 航点数:{len(ast_p)} | 路径长度:{path_length}m")
        return ast_p

# ==================== 通信日志全局函数 ====================
def init_comm_log():
    if "gcs2fcu_log" not in st.session_state:
        st.session_state.gcs2fcu_log = []
    if "fcu2gcs_log" not in st.session_state:
        st.session_state.fcu2gcs_log = []

def add_gcs_obc_fcu_log(msg):
    init_comm_log()
    t_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.gcs2fcu_log.append(f"[{t_str}] ✅ {msg}")

def add_fcu_obc_gcs_log(msg):
    init_comm_log()
    t_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    st.session_state.fcu2gcs_log.append(f"[{t_str}] {msg}")

# ==================== 障碍物管理 ====================
def save_obstacles_to_cache():
    if 'saved_obstacles' not in st.session_state:
        st.session_state.saved_obstacles = []
    st.session_state.saved_obstacles = copy.deepcopy(st.session_state.obstacles_gcj)
    st.success(f"已保存 {len(st.session_state.obstacles_gcj)} 个障碍物到缓存")

def load_obstacles_from_cache():
    if 'saved_obstacles' not in st.session_state or not st.session_state.saved_obstacles:
        st.warning("缓存中无障碍物，请先保存")
        return False
    st.session_state.obstacles_gcj = st.session_state.saved_obstacles
    st.success(f"已从缓存加载 {len(st.session_state.obstacles_gcj)} 个障碍物")
    return True

# ==================== 心跳包模拟器 ====================
class HeartbeatSimulator:
    def __init__(self, start_point_gcj):
        self.history = []
        self.current_pos = start_point_gcj.copy()
        self.path = [start_point_gcj.copy()]
        self.path_index = 0
        self.simulating = False
        self.paused = False
        self.flight_altitude = 50
        self.speed = 50
        self.progress = 0.0
        self.total_distance = 0.0
        self.distance_traveled = 0.0
        self.start_time = None
        self.wp_logged = set()

    def set_path(self, path, altitude=50, speed=50):
        self.path = path
        self.path_index = 0
        self.current_pos = path[0].copy()
        self.flight_altitude = altitude
        self.speed = speed
        self.simulating = True
        self.paused = False
        self.progress = 0.0
        self.distance_traveled = 0.0
        self.total_distance = 0.0
        self.start_time = datetime.now()
        self.wp_logged = set()
        add_fcu_obc_gcs_log("FCU→OBC→GCS: ACK | Mode: AUTO")
        for i in range(len(path)-1):
            self.total_distance += distance(path[i], path[i+1])

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.simulating = False
        self.paused = False

    def reset(self):
        self.path_index = 0
        self.current_pos = self.path[0].copy()
        self.progress = 0.0
        self.distance_traveled = 0.0
        self.start_time = None
        self.history = []
        self.wp_logged = set()

    def update_and_generate(self):
        if self.simulating and not self.paused and self.path_index < len(self.path)-1:
            target = self.path[self.path_index+1]
            dx = target[0] - self.current_pos[0]
            dy = target[1] - self.current_pos[1]
            dist_to_target = math.hypot(dx, dy)
            step = 0.00015 + (self.speed/100)*0.0005
            if dist_to_target < step:
                self.distance_traveled += dist_to_target
                self.current_pos = target.copy()
                wp_idx = self.path_index +1
                if wp_idx not in self.wp_logged:
                    add_fcu_obc_gcs_log(f"FCU→OBC→GCS: WP_REACHED #{wp_idx}")
                    self.wp_logged.add(wp_idx)
                self.path_index += 1
            else:
                ratio = step / dist_to_target
                self.current_pos[0] += dx * ratio
                self.current_pos[1] += dy * ratio
                self.distance_traveled += step
            if self.total_distance > 0:
                self.progress = min(1.0, self.distance_traveled / self.total_distance)
            if self.path_index >= len(self.path)-1:
                self.simulating = False
                self.progress = 1.0
                add_fcu_obc_gcs_log("FCU→OBC→GCS: MISSION_COMPLETE")
        else:
            self.simulating = False
            self.progress = 1.0
        altitude = self.flight_altitude + random.randint(-5,5) if self.simulating else random.randint(0,10)
        speed_display = round(self.speed * 0.1, 1) if self.simulating and not self.paused else 0
        elapsed_seconds = int((datetime.now() - self.start_time).total_seconds()) if self.start_time else 0
        remaining_distance_deg = self.total_distance - self.distance_traveled
        remaining_distance_m = remaining_distance_deg * 111000
        if speed_display > 0:
            remaining_time = int(remaining_distance_m / speed_display)
        else:
            remaining_time = 0
        battery = max(0, round(100 - (elapsed_seconds / 600) * 4, 0)) if self.simulating else 96
        data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "lng": self.current_pos[0],
            "lat": self.current_pos[1],
            "altitude": altitude,
            "voltage": round(random.uniform(11.5,12.8),1),
            "satellites": random.randint(8,14),
            "speed": speed_display,
            "progress": self.progress,
            "distance_traveled": self.distance_traveled,
            "total_distance": self.total_distance,
            "simulating": self.simulating,
            "paused": self.paused,
            "elapsed_time": elapsed_seconds,
            "remaining_distance": round(remaining_distance_m, 1),
            "remaining_time": remaining_time,
            "battery": int(battery),
            "current_waypoint": self.path_index + 1,
            "total_waypoints": len(self.path)
        }
        self.history.insert(0, data)
        if len(self.history) > 200:
            self.history.pop()
        return data

# ==================== 创建地图 ====================
def create_planning_map(center_gcj, points_gcj, obstacles_gcj, flight_history=None, planned_path=None, map_type="satellite", straight_blocked=True, safe_radius=5):
    if map_type == "satellite":
        tiles = GAODE_SATELLITE_URL_ALT
        attr = "高德卫星地图"
    else:
        tiles = GAODE_VECTOR_URL
        attr = "高德矢量地图"
    m = folium.Map(location=[center_gcj[1], center_gcj[0]], zoom_start=16, tiles=tiles, attr=attr)
    draw = plugins.Draw(
        export=True, position='topleft',
        draw_options={'polygon': {'allowIntersection': False, 'showArea': True, 'color': '#ff0000', 'fillColor': '#ff0000', 'fillOpacity': 0.4},
                      'polyline': False, 'rectangle': False, 'circle': False, 'marker': False, 'circlemarker': False},
        edit_options={'edit': True, 'remove': True}
    )
    m.add_child(draw)

    safe_offset = safe_radius / 111000.0
    for i, obs in enumerate(obstacles_gcj):
        poly = obs.get('polygon', [])
        if len(poly) < 3:
            continue
        for (x, y) in poly:
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                dx = math.cos(rad) * safe_offset
                dy = math.sin(rad) * safe_offset
                cx = x + dx
                cy = y + dy
                folium.CircleMarker(
                    location=[cy, cx],
                    radius=1.8,
                    color='#00ccff',
                    fill=True,
                    fill_color='#00ccff',
                    fill_opacity=0.7,
                    popup=f'安全半径 {safe_radius}m'
                ).add_to(m)
        coords = obs.get('polygon', [])
        if coords and len(coords) >= 3:
            popup_text = f"🚧 {obs.get('name', f'障碍物{i+1}')}\n高度: {obs.get('height', 20)}m"
            folium.Polygon([[c[1], c[0]] for c in coords], color="red", weight=3, fill=True, fill_color="red", fill_opacity=0.4, popup=popup_text).add_to(m)

    if points_gcj.get('A'):
        folium.Marker([points_gcj['A'][1], points_gcj['A'][0]], popup="🟢 起点", icon=folium.Icon(color="green", icon="play", prefix="fa")).add_to(m)
    if points_gcj.get('B'):
        folium.Marker([points_gcj['B'][1], points_gcj['B'][0]], popup="🔴 终点", icon=folium.Icon(color="red", icon="stop", prefix="fa")).add_to(m)
    if planned_path and len(planned_path) > 1:
        path_locations = [[p[1], p[0]] for p in planned_path]
        folium.PolyLine(path_locations, color="green", weight=5, opacity=0.9, popup="✈️ 智能避障航线").add_to(m)
        for point in planned_path:
            folium.CircleMarker([point[1], point[0]], radius=3, color="green", fill=True, fill_color="white", fill_opacity=0.8).add_to(m)
    if points_gcj.get('A') and points_gcj.get('B'):
        if not straight_blocked:
            folium.PolyLine([[points_gcj['A'][1], points_gcj['A'][0]], [points_gcj['B'][1], points_gcj['B'][0]]], color="blue", weight=2, opacity=0.5, dash_array='5, 5', popup="直线航线").add_to(m)
        else:
            folium.PolyLine([[points_gcj['A'][1], points_gcj['A'][0]], [points_gcj['B'][1], points_gcj['B'][0]]], color="gray", weight=2, opacity=0.4, dash_array='5, 5', popup="⚠️ 直线被阻挡").add_to(m)
    if flight_history and len(flight_history) > 1:
        trail = [[p[1], p[0]] for p in flight_history if len(p) >= 2]
        if len(trail) > 1:
            folium.PolyLine(trail, color="orange", weight=2, opacity=0.6, popup="历史轨迹").add_to(m)
    return m

# ==================== 主程序 ====================
def main():
    init_comm_log()
    st.title("🏫 无人机地面站系统 - 平行偏移绕行")
    st.markdown("---")

    if "points_gcj" not in st.session_state:
        st.session_state.points_gcj = {'A': DEFAULT_A_GCJ.copy(), 'B': DEFAULT_B_GCJ.copy()}
    if "obstacles_gcj" not in st.session_state:
        st.session_state.obstacles_gcj = []
    if "saved_obstacles" not in st.session_state:
        st.session_state.saved_obstacles = []
    if "heartbeat_sim" not in st.session_state:
        st.session_state.heartbeat_sim = HeartbeatSimulator(st.session_state.points_gcj['A'].copy())
    if "last_hb_time" not in st.session_state:
        st.session_state.last_hb_time = time.time()
    if "simulation_running" not in st.session_state:
        st.session_state.simulation_running = False
    if "flight_altitude" not in st.session_state:
        st.session_state.flight_altitude = 50
    if "flight_history" not in st.session_state:
        st.session_state.flight_history = []
    if "planned_path" not in st.session_state:
        st.session_state.planned_path = None
    if "pending_polygon" not in st.session_state:
        st.session_state.pending_polygon = None
    if "pending_height" not in st.session_state:
        st.session_state.pending_height = 20

    st.sidebar.title("🎛️ 导航菜单")
    page = st.sidebar.radio("选择功能模块", ["🗺️ 航线规划", "📡 飞行监控", "🚧 障碍物管理"])
    map_type_choice = st.sidebar.radio("🗺️ 地图类型", ["卫星影像", "矢量街道"], index=0)
    map_type = "satellite" if map_type_choice == "卫星影像" else "vector"

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ 无人机参数")
    drone_speed = st.sidebar.slider("飞行速度系数", min_value=10, max_value=100, value=50, step=5)
    safe_radius = st.sidebar.number_input("安全半径 (米)", min_value=1, max_value=30, value=5, step=1)
    flight_alt = st.sidebar.number_input("飞行高度 (米)", min_value=0, max_value=200, value=st.session_state.flight_altitude, step=5)
    st.session_state.flight_altitude = flight_alt

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 绕行策略")
    strategy = st.sidebar.radio("选择避障方式", ["最佳航线 (A*)", "向左绕行", "向右绕行"], index=0)
    strategy_map = {"最佳航线 (A*)": "best", "向左绕行": "left", "向右绕行": "right"}
    selected_strategy = strategy_map[strategy]

    st.sidebar.markdown("---")
    obs_count = len(st.session_state.obstacles_gcj)
    straight_blocked = is_path_blocked(
        st.session_state.points_gcj['A'],
        st.session_state.points_gcj['B'],
        st.session_state.obstacles_gcj,
        st.session_state.flight_altitude
    )
    st.sidebar.info(f"🏫 校园区域\n🚧 障碍物: {obs_count}\n📌 直线: {'🚫 被阻挡' if straight_blocked else '✅ 畅通'}")

    if st.sidebar.button("🔄 刷新数据", use_container_width=True):
        st.session_state.planned_path = create_avoidance_path(
            st.session_state.points_gcj['A'],
            st.session_state.points_gcj['B'],
            st.session_state.obstacles_gcj,
            st.session_state.flight_altitude,
            safe_radius,
            selected_strategy
        )
        st.rerun()

    # ==================== 航线规划 ====================
    if page == "🗺️ 航线规划":
        st.header("🗺️ 航线规划 - 智能避障")
        if straight_blocked:
            st.warning(f"⚠️ 直线航线被建筑物阻挡！障碍物高度 > 当前飞行高度 {flight_alt}m")
        else:
            st.success(f"✅ 直线航线畅通无阻 (飞行高度 {flight_alt}m)")

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader("🎮 控制面板")
            st.markdown("#### 🟢 起点 A")
            a_lat = st.number_input("纬度", value=st.session_state.points_gcj['A'][1], format="%.6f", key="a_lat")
            a_lng = st.number_input("经度", value=st.session_state.points_gcj['A'][0], format="%.6f", key="a_lng")
            if st.button("📍 设置 A 点", use_container_width=True):
                st.session_state.points_gcj['A'] = [a_lng, a_lat]
                st.session_state.planned_path = create_avoidance_path(
                    st.session_state.points_gcj['A'], st.session_state.points_gcj['B'],
                    st.session_state.obstacles_gcj, st.session_state.flight_altitude, safe_radius, selected_strategy
                )
                st.rerun()

            st.markdown("#### 🔴 终点 B")
            b_lat = st.number_input("纬度", value=st.session_state.points_gcj['B'][1], format="%.6f", key="b_lat")
            b_lng = st.number_input("经度", value=st.session_state.points_gcj['B'][0], format="%.6f", key="b_lng")
            if st.button("📍 设置 B 点", use_container_width=True):
                st.session_state.points_gcj['B'] = [b_lng, b_lat]
                st.session_state.planned_path = create_avoidance_path(
                    st.session_state.points_gcj['A'], st.session_state.points_gcj['B'],
                    st.session_state.obstacles_gcj, st.session_state.flight_altitude, safe_radius, selected_strategy
                )
                st.rerun()

            st.markdown("#### 🏗️ 新障碍物高度")
            new_obs_height = st.number_input("高度 (米)", min_value=1, max_value=200, value=st.session_state.pending_height, step=5)
            st.session_state.pending_height = new_obs_height

            if st.button("➕ 添加障碍物（从当前圈选）", use_container_width=True):
                if st.session_state.pending_polygon and len(st.session_state.pending_polygon) >= 3:
                    st.session_state.obstacles_gcj.append({
                        "name": f"建筑物{len(st.session_state.obstacles_gcj)+1}",
                        "polygon": st.session_state.pending_polygon,
                        "height": st.session_state.pending_height
                    })
                    st.success(f"已添加障碍物（高度{st.session_state.pending_height}m）")
                    st.session_state.pending_polygon = None
                    st.session_state.planned_path = create_avoidance_path(
                        st.session_state.points_gcj['A'], st.session_state.points_gcj['B'],
                        st.session_state.obstacles_gcj, st.session_state.flight_altitude, safe_radius, selected_strategy
                    )
                    st.rerun()
                else:
                    st.warning("请先在地图绘制多边形")

            if st.button("🔄 重新规划路径", use_container_width=True):
                st.session_state.planned_path = create_avoidance_path(
                    st.session_state.points_gcj['A'], st.session_state.points_gcj['B'],
                    st.session_state.obstacles_gcj, st.session_state.flight_altitude, safe_radius, selected_strategy
                )
                st.rerun()

            st.markdown("#### ✈️ 飞行控制")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶️ 开始飞行", use_container_width=True):
                    path = st.session_state.planned_path or [st.session_state.points_gcj['A'], st.session_state.points_gcj['B']]
                    st.session_state.heartbeat_sim.set_path(path, st.session_state.flight_altitude, drone_speed)
                    st.session_state.simulation_running = True
                    st.session_state.flight_history = []
                    st.success("已开始飞行")
            with c2:
                if st.button("⏹️ 停止飞行", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.session_state.heartbeat_sim.stop()

        with col2:
            st.subheader("🗺️ 规划地图")
            center = st.session_state.points_gcj['A'] or SCHOOL_CENTER_GCJ
            if st.session_state.planned_path is None:
                st.session_state.planned_path = create_avoidance_path(
                    st.session_state.points_gcj['A'], st.session_state.points_gcj['B'],
                    st.session_state.obstacles_gcj, st.session_state.flight_altitude, safe_radius, selected_strategy
                )
            m = create_planning_map(center, st.session_state.points_gcj, st.session_state.obstacles_gcj,
                                   st.session_state.flight_history, st.session_state.planned_path, map_type, straight_blocked, safe_radius)
            output = st_folium(m, width=700, height=550, returned_objects=["last_active_drawing"])

            if output and output.get("last_active_drawing"):
                last = output["last_active_drawing"]
                if last and last.get("geometry") and last["geometry"]["type"] == "Polygon":
                    coords = last["geometry"]["coordinates"]
                    if coords:
                        poly = [[p[0], p[1]] for p in coords[0]]
                        if len(poly) >= 3:
                            st.session_state.pending_polygon = poly
                            st.success("已捕获多边形")

    # ==================== 飞行监控 ====================
    elif page == "📡 飞行监控":
        st.header("🛸 飞行实时画面 - 任务执行监控")

        col_ctrl, col_status = st.columns([3, 1])
        with col_ctrl:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("开始任务", type="primary", use_container_width=True):
                    path = st.session_state.planned_path or [st.session_state.points_gcj['A'], st.session_state.points_gcj['B']]
                    st.session_state.heartbeat_sim.set_path(path, st.session_state.flight_altitude, drone_speed)
                    st.session_state.simulation_running = True
                    st.session_state.flight_history = []
                    st.rerun()
            with c2:
                if st.button("暂停", use_container_width=True):
                    st.session_state.heartbeat_sim.pause()
                    st.rerun()
            with c3:
                if st.button("停止", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.session_state.heartbeat_sim.stop()
                    st.rerun()
            with c4:
                if st.button("重置", use_container_width=True):
                    st.session_state.heartbeat_sim.reset()
                    st.session_state.simulation_running = False
                    st.session_state.flight_history = []
                    st.rerun()
        with col_status:
            status = "运行中" if st.session_state.simulation_running and not st.session_state.heartbeat_sim.paused else "已暂停"
            st.info(f"状态：{status}")

        current_time = time.time()
        auto_refresh = False
        if st.session_state.simulation_running and not st.session_state.heartbeat_sim.paused:
            if current_time - st.session_state.last_hb_time >= 0.15:
                sim_data = st.session_state.heartbeat_sim.update_and_generate()
                pos_gcj = [sim_data["lng"], sim_data["lat"]]
                st.session_state.flight_history.append(pos_gcj)
                st.session_state.last_hb_time = current_time
                auto_refresh = True
        
        if auto_refresh:
            st.rerun()

        if st.session_state.heartbeat_sim.history:
            latest = st.session_state.heartbeat_sim.history[0]
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("当前航点", f"{latest['current_waypoint']}/{latest['total_waypoints']}")
            col2.metric("飞行速度", f"{latest['speed']} m/s")
            col3.metric("已用时间", f"{timedelta(seconds=latest['elapsed_time'])}")
            col4.metric("剩余距离", f"{latest['remaining_distance']} m")
            col5.metric("预计到达", str(timedelta(seconds=latest['remaining_time'])) if latest['remaining_time']>0 else "00:00")
            col6.metric("电量模拟", f"{latest['battery']}%")

            st.progress(latest['progress'], text=f"任务进度：{latest['progress']*100:.0f}%")
            st.markdown("---")

            map_col, comm_col = st.columns([2, 1])
            with map_col:
                st.subheader("实时飞行地图")
                center = st.session_state.points_gcj['A'] or SCHOOL_CENTER_GCJ
                m = create_planning_map(center, st.session_state.points_gcj, st.session_state.obstacles_gcj,
                                       st.session_state.flight_history, st.session_state.planned_path, map_type, straight_blocked, safe_radius)
                folium_static(m, width=600, height=400)
            with comm_col:
                st.subheader("📡 通信链路拓扑与数据流")
                topo_html = '''
                <div style="display:flex; justify-content:space-around; text-align:center; margin-top:10px;">
                    <div style="width:28%; padding:12px; background:#e3f2fd; border:2px solid #1976d2; border-radius:8px;">
                        <div style="font-weight:bold; font-size:16px; color:#1976d2;">GCS</div>
                        <div style="font-size:12px;">地面站<br/>192.168.1.100</div>
                        <div style="color:green; font-size:13px;">✅在线</div>
                    </div>
                    <div style="display:flex;align-items:center;">⬇️UDP:14550⬆️</div>
                    <div style="width:28%; padding:12px; background:#fff8e1; border:2px solid #f57c00; border-radius:8px;">
                        <div style="font-weight:bold; font-size:16px; color:#f57c00;">OBC</div>
                        <div style="font-size:12px;">机载计算机<br/>Raspberry Pi4</div>
                        <div style="color:green; font-size:13px;">✅在线</div>
                    </div>
                    <div style="display:flex;align-items:center;">⬇️MAVLink⬆️</div>
                    <div style="width:28%; padding:12px; background:#fce4ec; border:2px solid #c2185b; border-radius:8px;">
                        <div style="font-weight:bold; font-size:16px; color:#c2185b;">FCU</div>
                        <div style="font-size:12px;">飞控<br/>PX4/ArduPilot</div>
                        <div style="color:green; font-size:13px;">✅在线</div>
                    </div>
                </div>
                <div style="margin-top:15px; padding:8px; background:#f5f5f5; border-radius:6px; font-size:13px;">
                📊链路统计：GCS↔OBC:正常｜OBC↔FCU:正常｜延迟:~25ms｜丢包率:0.1%
                </div>
                '''
                st.markdown(topo_html, unsafe_allow_html=True)

                tab1, tab2 = st.tabs(["📤GCS→OBC→FCU下发日志", "📥FCU→OBC→GCS回传日志"])
                with tab1:
                    log_text1 = ""
                    if len(st.session_state.gcs2fcu_log) == 0:
                        log_text1 = "暂无航线下发日志\n点击重新规划生成日志"
                    else:
                        for line in st.session_state.gcs2fcu_log[-30:]:
                            log_text1 += line + "\n"
                    st.text_area("", log_text1, height=220)
                with tab2:
                    log_text2 = ""
                    if len(st.session_state.fcu2gcs_log) == 0:
                        log_text2 = "暂无飞控回传日志\n启动飞机飞行生成抵达日志"
                    else:
                        for line in st.session_state.fcu2gcs_log[-30:]:
                            log_text2 += line + "\n"
                    st.text_area("", log_text2, height=220)

    # ==================== 障碍物管理 ====================
    elif page == "🚧 障碍物管理":
        st.header("🚧 障碍物管理")
        st.info(f"当前共 {len(st.session_state.obstacles_gcj)} 个障碍物")
        col1, col2 = st.columns([1, 1.5])
        with col1:
            for i, obs in enumerate(st.session_state.obstacles_gcj):
                na, h, btn = st.columns([2,1,1])
                na.write(f"🚧 {obs.get('name', f'障碍物{i+1}')}")
                h.write(f"{obs.get('height',20)}m")
                if btn.button("删除", key=f"del{i}"):
                    st.session_state.obstacles_gcj.pop(i)
                    st.rerun()
            st.columns(2)[0].button("💾 保存到缓存", on_click=save_obstacles_to_cache)
            st.columns(2)[1].button("📂 从缓存加载", on_click=load_obstacles_from_cache)
            if st.button("🗑️ 全部清除"):
                st.session_state.obstacles_gcj = []
                st.rerun()

if __name__ == "__main__":
    main()


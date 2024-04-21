from __future__ import annotations

from common.astar import AStar
from minigrid.core.constants import *
from typing import Union, Iterable
from itertools import islice
import numpy as np


class _MiniGridAStar(AStar):

    def __init__(self, memory_obs: np.ndarray, goal: (int, int) = None, start: (int, int) = None):
        self.memory_obs = memory_obs
        self.goal = goal
        self.start = start

    def heuristic_cost_estimate(self, current, goal) -> float:
        """
        计算启发式距离，使用曼哈顿距离
        :param current:
        :param goal:
        :return:
        """
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def distance_between(self, n1, n2) -> float:
        """
        计算两个节点之间的距离, 使用曼哈顿距离
        n2 is guaranteed to belong to the list returned by the call to neighbors(n1).
        :param n1:
        :param n2:
        :return:
        """
        distance = manhattan_distance(n1, n2)

        for pos in [n1, n2]:
            if pos == self.goal:
                return distance

        for pos in [n1, n2]:
            obj_idx, _, state = self.memory_obs[pos[0], pos[1], :]
            obj = IDX_TO_OBJECT[obj_idx]
            if obj == 'key' and pos != self.start:
                # 不能穿过墙壁、球、箱子、岩浆
                return float('inf')

            if obj in ['wall', 'ball', 'box', 'lava', 'agent']:
                # 不能穿过墙壁、球、箱子、岩浆
                # print('不能穿过墙壁、球、箱子、岩浆',
                #       f'pos={pos} start={self.start} goal={self.goal} obj={obj} state={state}')
                return float('inf')

            if obj == 'door' and state != STATE_TO_IDX['open']:
                # 不能穿过关闭的门
                return float('inf')

        return distance

    def neighbors(self, node):
        """
        返回当前节点的邻居节点
        :param node:
        :return:
        """
        for dx, dy in DIR_TO_VEC:
            x2 = node[0] + dx
            y2 = node[1] + dy
            if x2 < 0 or x2 >= self.memory_obs.shape[0] or y2 < 0 or y2 >= self.memory_obs.shape[1]:
                continue
            yield x2, y2


# def is_match_obs(obs: (int, int, int), target: (int, int, int)) -> int:
#     """
#     将object转换为对应的idx
#     :param object:
#     :return:
#     """
#     target_object, target_color, target_state = target
#     obs_object, obs_color, obs_state = obs
#


# 找到离自己最近的物体
def find_nearest_object_pos(
        obj: str,
        memory_obs: np.ndarray,
        agent_pos: (int, int), color: str = '',
        near_range: (int, int) = (0, 1e6)) -> (int, int):
    """
    找到离自己最近的物体
    :param obj:
    :param memory_obs:
    :param agent_pos:
    :param color:
    :param near_range: [min_distance, max_distance]
    :return:
    """
    min_distance = 1e6
    door_pos = None
    for x in range(memory_obs.shape[0]):
        for y in range(memory_obs.shape[1]):
            distance = abs(x - agent_pos[0]) + abs(y - agent_pos[1])
            if distance < near_range[0] or distance > near_range[1]:
                continue
            object_idx, color_idx, state = memory_obs[x, y, :]
            if object_idx == OBJECT_TO_IDX[obj] and (color == '' or color_idx == COLOR_TO_IDX[color]):
                if distance < min_distance:
                    min_distance = distance
                    door_pos = (x, y)

    return door_pos


# 曼哈顿距离
def manhattan_distance(pos1: (int, int), pos2: (int, int)) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def astar_find_path(obs, start, target) -> Union[Iterable[(int, int)], None]:
    """
    使用A*算法寻找路径
    :param obs:
    :param start:
    :param target:
    :return:
    """
    astar = _MiniGridAStar(memory_obs=obs, goal=target, start=start)
    path = astar.astar(start, target)
    return path


def iter_take(iterable, n):
    """
    从迭代器中取出n个元素
    :param iterable: 迭代器
    :param n: 取出的元素个数
    :return:
    """
    return list(islice(iterable, n))


import uuid
from typing import Union, Tuple
import xml.etree.ElementTree as ET
import os
import json

import gym.core
import numpy as np


def new_node_id():
    # 只取uuid的前10位
    return uuid.uuid4().hex[:10]


def camel_case_to_snake_case(name):
    """
    驼峰转蛇形
    :param name:
    :return:
    """
    return ''.join(['_' + i.lower() if i.isupper() else i for i in name]).lstrip('_')


def parse_prop_options(options: Union[list, dict, str]) -> list:
    """
    解析属性选项
    :param options:
    :return:
    """
    if isinstance(options, str):
        options = options.split(',')
    if isinstance(options, list):
        result = []
        for option in options:
            if isinstance(option, str):
                result.append({
                    'name' : option,
                    'value': option
                })
            elif isinstance(option, dict):
                result.append({
                    'name' : option.get('name', ''),
                    'value': option.get('value', '')
                })
        return result
    elif isinstance(options, dict):
        result = []
        for key in options:
            result.append({
                'name' : key,
                'value': options[key]
            })
        return result
    return []


PROP_TYPE_MAPPER = {
    'str'   : str,
    'string': str,
    'int'   : int,
    'float' : float,
    'double': float,
    'number': float,
    'bool'  : bool,
    'list'  : list,
    'dict'  : dict,
    'json'  : dict,
}


def parse_prop_type(prop_type: [str, type]):
    if isinstance(prop_type, str):
        prop_type = prop_type.lower()
        if prop_type in PROP_TYPE_MAPPER:
            return PROP_TYPE_MAPPER[prop_type]
        else:
            return str
    else:
        return prop_type


def parse_type_value(value, value_type):
    value_type = parse_prop_type(value_type)
    if value_type == bool:
        return parse_bool_value(value)
    elif value_type == int:
        return parse_int_value(value)
    elif value_type == float:
        return parse_float_value(value)
    elif value_type == str:
        return str(value)
    elif value_type == list:
        return parse_list_value(value)
    elif value_type == dict:
        return parse_dict_value(value)
    elif callable(value_type):
        return value_type(value)
    return value


# 最终props都是以列表的形式保存的
def parse_props(props):
    if props is None:
        return []
    result = []
    if isinstance(props, list):
        for prop in props:
            if isinstance(prop, str):
                result.append({
                    'name'    : prop,
                    'type'    : 'str',
                    'default' : '',
                    'required': False,
                    'desc'    : '',
                    'options' : None,  # 选项 用于下拉框 仅在type为str时有效 {'name': '选项1', 'value': '1'}
                    'visible' : True,  # 是否可见
                })
            elif isinstance(prop, dict):
                result.append({
                    'name'    : prop.get('name', ''),
                    'type'    : prop.get('type', 'str'),
                    'default' : prop.get('default', ''),
                    'required': prop.get('required', False),
                    'desc'    : prop.get('desc', ''),
                    'options' : prop.get('options', None),
                    'visible' : prop.get('visible', True),
                })
    elif isinstance(props, dict):
        for prop in props:
            prop_item = props[prop]
            if isinstance(prop_item, dict):
                result.append({
                    'name'    : prop,
                    'type'    : prop_item.get('type', 'str'),
                    'default' : prop_item.get('default', ''),
                    'required': prop_item.get('required', False),
                    'desc'    : prop_item.get('desc', ''),
                    'options' : prop_item.get('options', None),
                    'visible' : prop_item.get('visible', True),
                })
            elif isinstance(prop_item, type):
                result.append({
                    'name'    : prop,
                    'type'    : prop_item,
                    'default' : '',
                    'required': False,
                    'desc'    : '',
                    'options' : None,
                    'visible' : True,
                })

    for i, item in enumerate(result):
        result[i]['type'] = parse_prop_type(item['type']).__name__
        if not callable(item['default']):
            result[i]['default'] = parse_type_value(value=item['default'], value_type=item['type'])
        result[i]['options'] = parse_prop_options(item['options'])

    return result


def merge_props(props: list, to_props: list):
    """
    合并两个props
    :param props:
    :param to_props:
    :return:
    """
    if to_props is None:
        return props
    to_props = to_props.copy()
    for prop in props:
        find_index = find_prop_index(to_props, prop['name'])
        if find_index == -1:
            to_props.append(prop)
        else:
            to_props[find_index] = prop
    return to_props


def find_prop(meta, name):
    if 'props' in meta:
        for prop in meta['props']:
            if prop['name'] == name:
                return prop
    return None


def find_prop_index(props, name):
    for index, prop in enumerate(props):
        if prop['name'] == name:
            return index
    return -1


def parse_bool_value(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, int):
        return value > 0
    elif isinstance(value, float):
        return value > 0.0
    elif isinstance(value, str):
        return value.lower() in ['true', '1', 'yes', 'y']
    return False


def parse_int_value(value: str, default: int = 0):
    try:
        return int(value)
    except:
        return default


def parse_float_value(value: str, default: float = 0.0):
    try:
        return float(value)
    except:
        return default


def parse_list_value(value: str, default: list = None):
    try:
        value = json.loads(value)
        return value
    except:
        return default


def parse_dict_value(value: str, default: dict = None):
    try:
        value = json.loads(value)
        return value
    except:
        return default


# 定义一个函数将 XML 元素转换为字典
def xml_to_dict(element):
    result = {
        'tag'       : element.tag,
        'attributes': element.attrib,
        'children'  : [xml_to_dict(child) for child in element]
    }
    return result


def read_xml_to_dict(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return xml_to_dict(root)


# 从目录中提取出所有的xml文件
def extract_xml_files_from_dir(dir_path: str):
    xml_files = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.xml'):
                xml_files.append(os.path.join(root, file))
    return xml_files


def is_obs_same(obs: gym.core.ObsType, other: gym.core.ObsType) -> bool:
    if isinstance(obs, np.ndarray):
        return (obs == other).all()

    if isinstance(obs, gym.core.Dict):
        for key, value in obs.items():
            if not is_obs_same(value, other[key]):
                return False
        return True

    try:
        return obs == other
    except Exception as e:
        print('is_obs_same', e, obs, other)
        return False

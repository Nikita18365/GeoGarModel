"""Bounded, elementwise expressions for the voxel property editor."""
import ast
from functools import reduce

import numpy as np

from .model_archive import regular_cells
from .work_progress import report


BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
          ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.remainder,
          ast.BitAnd: np.logical_and, ast.BitOr: np.logical_or}
COMPARE = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
           ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
UNARY = {ast.UAdd: np.positive, ast.USub: np.negative, ast.Not: np.logical_not,
         ast.Invert: np.logical_not}
FUNCTIONS = {'where': (np.where,3), 'abs': (np.abs,1), 'sqrt': (np.sqrt,1),
             'min': (np.minimum,2), 'max': (np.maximum,2), 'clip': (np.clip,3)}


class Formula:
    def __init__(self, text):
        self.text = text.strip()
        self.names = set()
        if len(self.text)>1024:
            raise ValueError('Максимальная длина формулы — 1024 знака.')
        try:
            tree = ast.parse(self.text, mode='eval')
        except SyntaxError as exc:
            raise ValueError('Проверьте скобки, операторы и десятичные точки в формуле.') from exc
        if sum(1 for _ in ast.walk(tree))>160:
            raise ValueError('Слишком сложная формула.')
        self.evaluate = self.compile(tree.body)

    def compile(self, node):
        if isinstance(node, ast.Name) and node.id in ('A','B','X','Y','Z'):
            self.names.add(node.id)
            return lambda env: env[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (float,int):
            try:
                value = float(node.value)
            except OverflowError as exc:
                raise ValueError('Числовая константа слишком велика.') from exc
            if not np.isfinite(value):
                raise ValueError('В формуле допускаются только конечные константы.')
            return lambda env: value
        if isinstance(node, ast.Compare):
            args = [self.compile(arg) for arg in [node.left,*node.comparators]]
            ops = [COMPARE.get(type(op)) for op in node.ops]
            if all(ops):
                def compare(env):
                    values = [arg(env) for arg in args]
                    return reduce(np.logical_and,(op(a,b) for op,a,b in zip(ops,values,values[1:])))
                return compare
        if isinstance(node, ast.BoolOp):
            op = np.logical_and if isinstance(node.op,ast.And) else np.logical_or
            args = [self.compile(arg) for arg in node.values]
            return lambda env: reduce(op,(arg(env) for arg in args))
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            func, children = UNARY[type(node.op)], [node.operand]
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            func, children = BINARY[type(node.op)], [node.left,node.right]
        elif isinstance(node, ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCTIONS:
            func, count = FUNCTIONS[node.func.id]
            children = node.args
            if node.keywords or len(children)!=count:
                raise ValueError(f'Функция {node.func.id} принимает {count} арг.')
        else:
            raise ValueError('Неизвестное имя или недопустимая операция в формуле.')
        args = [self.compile(arg) for arg in children]
        return lambda env: func(*(arg(env) for arg in args))


def calculate(a, formula, b=None):
    mesh = a.mesh
    if 'B' in formula.names:
        if b is None or b.mode!=a.mode or (a.crs and b.crs and a.crs!=b.crs):
            raise ValueError('Для B требуется модель с тем же свойством и совместимой СК.')
        if any(not np.array_equal(getattr(mesh,k),getattr(b.mesh,k)) for k in ('east','north','depth')):
            raise ValueError('Сетки A и B различаются. Перенесите модели на общую сетку.')
    result = a.copy()
    coords = formula.names.intersection('XYZ')
    for start in range(0,mesh.size,65536):
        end = min(mesh.size,start+65536)
        ids = start+np.flatnonzero(mesh.active[start:end])
        if not len(ids):
            report(end,mesh.size,'Вычисление формулы')
            continue
        env = {'A':a.values[ids]}
        cells = None
        if 'B' in formula.names:
            if not b.mesh.active[ids].all():
                raise ValueError('В B отсутствуют активные ячейки, используемые в A.')
            if mesh.bounds is not None or b.mesh.bounds is not None:
                cells = mesh.bounds[ids] if mesh.bounds is not None else regular_cells(mesh,ids)
                other = b.mesh.bounds[ids] if b.mesh.bounds is not None else regular_cells(b.mesh,ids)
                if not np.array_equal(cells,other):
                    raise ValueError('Обрезанные границы активных ячеек A и B различаются.')
            env['B'] = b.values[ids]
        if coords:
            if cells is None:
                cells = regular_cells(mesh,ids) if mesh.bounds is None else mesh.bounds[ids]
            for name in coords:
                i = 'XYZ'.index(name)
                center = (cells[:,i]+cells[:,i+3])/2
                env[name] = center[:,None] if a.values.ndim==2 else center
        with np.errstate(all='ignore'):
            values = formula.evaluate(env)
        if not np.isfinite(values).all():
            raise ValueError('Формула дала неопределённое или бесконечное значение. Проверьте область определения.')
        result.values[ids] = values
        report(end,mesh.size,'Вычисление формулы')
    result.history.append(dict(operation='formula',expression=formula.text,operand_b=b.source_path if b else ''))
    return result

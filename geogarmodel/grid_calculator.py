"""Safe elementwise grid arithmetic with explicit spatial alignment."""
import ast
from functools import reduce

import numpy as np
from pyproj import CRS

from .grids import Grid
from .work_progress import report


OPERATIONS = {
    'SUM': 'Сложение',
    'SUB': 'Вычитание',
    'MUL': 'Умножение',
    'DIV': 'Деление',
    'MEAN': 'Среднее',
    'MIN': 'Минимум',
    'MAX': 'Максимум',
    'AND': 'Логическое И',
    'OR': 'Логическое ИЛИ',
    'XOR': 'Исключающее ИЛИ',
    'NOT': 'Логическое НЕ',
    'ABS': 'Абсолютное значение',
    'SQRT': 'Квадратный корень',
    'LOG10': 'Десятичный логарифм',
    'FORMULA': 'Произвольная формула',
}
UNARY_OPERATIONS = frozenset(('NOT', 'ABS', 'SQRT', 'LOG10'))
LOGICAL_OPERATIONS = frozenset(('AND', 'OR', 'XOR', 'NOT'))
SAME_UNIT_OPERATIONS = frozenset(('SUM', 'SUB', 'MEAN', 'MIN', 'MAX', 'ABS'))


def operand_name(index):
    """Zero-based Excel-like labels A, B, ..., Z, AA, AB, ... ."""
    if index < 0:
        raise ValueError('Отрицательный номер операнда.')
    result = ''
    index += 1
    while index:
        index, rest = divmod(index-1, 26)
        result = chr(65+rest)+result
    return result


def _finite(value):
    value = np.asarray(value, dtype=float)
    return np.where(np.isfinite(value), value, np.nan)


def _logical(function, *values):
    valid = reduce(np.logical_and, (np.isfinite(v) for v in values))
    return np.where(valid, function(*values), np.nan)


def _where(condition, yes, no):
    return np.where(np.isfinite(condition), np.where(condition, yes, no), np.nan)


def _reduce(function, *values):
    return reduce(function, values)


def _mean(*values):
    return _reduce(np.add, *values)/len(values)


BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
          ast.Div: np.divide, ast.Pow: np.power, ast.Mod: np.remainder,
          ast.BitAnd: lambda a,b: _logical(np.logical_and,a,b),
          ast.BitOr: lambda a,b: _logical(np.logical_or,a,b),
          ast.BitXor: lambda a,b: _logical(np.logical_xor,a,b)}
COMPARE = {ast.Lt: np.less, ast.LtE: np.less_equal, ast.Gt: np.greater,
           ast.GtE: np.greater_equal, ast.Eq: np.equal, ast.NotEq: np.not_equal}
UNARY = {ast.UAdd: np.positive, ast.USub: np.negative,
         ast.Not: lambda a: _logical(np.logical_not,a),
         ast.Invert: lambda a: _logical(np.logical_not,a)}
FUNCTIONS = {
    'abs': (np.abs,1,1), 'sqrt': (np.sqrt,1,1), 'log': (np.log,1,1),
    'log10': (np.log10,1,1), 'exp': (np.exp,1,1), 'sin': (np.sin,1,1),
    'cos': (np.cos,1,1), 'clip': (np.clip,3,3), 'where': (_where,3,3),
    'min': (lambda *v: _reduce(np.minimum,*v),1,None),
    'max': (lambda *v: _reduce(np.maximum,*v),1,None),
    'sum': (lambda *v: _reduce(np.add,*v),1,None),
    'mean': (_mean,1,None),
}


class GridFormula:
    """Compile a whitelist of AST nodes; never evaluate Python source."""
    def __init__(self, expression, names):
        self.text = expression.strip()
        self.names = set()
        self.allowed = frozenset(names)
        if not self.text or len(self.text)>4096:
            raise ValueError('Введите формулу длиной до 4096 знаков.')
        try:
            tree = ast.parse(self.text, mode='eval')
            if sum(1 for _ in ast.walk(tree))>512:
                raise ValueError('Слишком сложная формула (более 512 узлов).')
            self.evaluate = self.compile(tree.body)
        except (SyntaxError, RecursionError) as exc:
            raise ValueError('Проверьте синтаксис, скобки и десятичные точки формулы.') from exc
        if not self.names:
            raise ValueError('Формула должна использовать хотя бы одну карту.')

    def compile(self, node):
        if isinstance(node, ast.Name) and node.id in self.allowed:
            self.names.add(node.id)
            return lambda env: env[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int,float):
            try:
                value = float(node.value)
            except OverflowError as exc:
                raise ValueError('Слишком большая константа.') from exc
            if not np.isfinite(value):
                raise ValueError('Константы должны быть конечными числами.')
            return lambda env: value
        if isinstance(node, ast.Compare):
            ops = [COMPARE.get(type(op)) for op in node.ops]
            if all(ops):
                args = [self.compile(arg) for arg in [node.left,*node.comparators]]
                def evaluate(env):
                    values = [arg(env) for arg in args]
                    compared = [_logical(op,a,b) for op,a,b in zip(ops,values,values[1:])]
                    return reduce(lambda a,b: _logical(np.logical_and,a,b),compared)
                return evaluate
        if isinstance(node, ast.BoolOp) and type(node.op) in (ast.And,ast.Or):
            op = np.logical_and if isinstance(node.op,ast.And) else np.logical_or
            args = [self.compile(arg) for arg in node.values]
            return lambda env: reduce(lambda a,b: _logical(op,a,b),(arg(env) for arg in args))
        if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
            function, children = UNARY[type(node.op)], [node.operand]
        elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
            function, children = BINARY[type(node.op)], [node.left,node.right]
        elif isinstance(node, ast.Call) and isinstance(node.func,ast.Name) and node.func.id in FUNCTIONS:
            function, minimum, maximum = FUNCTIONS[node.func.id]
            children = node.args
            if node.keywords or len(children)<minimum or (maximum is not None and len(children)>maximum):
                raise ValueError(f'Неверное число аргументов функции {node.func.id}.')
        else:
            raise ValueError('Неизвестное имя или недопустимая операция в формуле.')
        args = [self.compile(arg) for arg in children]
        return lambda env: _finite(function(*(arg(env) for arg in args)))


def suggested_unit(grids, operation):
    if operation in LOGICAL_OPERATIONS:
        return '1'
    units = {grid.unit for grid in grids}
    return next(iter(units)) if operation in SAME_UNIT_OPERATIONS and len(units)==1 else ''


def _same_crs(a,b):
    if bool(a.crs)!=bool(b.crs):
        return False
    if a.crs and not CRS.from_user_input(a.crs).equals(CRS.from_user_input(b.crs),ignore_axis_order=True):
        return False
    return (a.coordinate_unit==b.coordinate_unit or
            bool(a.crs) and 'unknown' in (a.coordinate_unit,b.coordinate_unit))


def _axis_equal(a,b):
    if a.shape!=b.shape:
        return False
    # Do not use relative tolerance against large projected eastings.
    step = min(a[1]-a[0],b[1]-b[0])
    return np.allclose(a,b,rtol=0,atol=step*1e-8)


class _Sampler:
    """Bilinear interpolation with no extrapolation and no NaN bridging."""
    def __init__(self, base, other, alignment, name):
        self.grid = other
        if not _same_crs(base,other):
            raise ValueError(f'СК или единицы координат A и {name} различаются.')
        self.direct = _axis_equal(base.x,other.x) and _axis_equal(base.y,other.y)
        if not self.direct and alignment=='strict':
            raise ValueError(f'Сетки A и {name} различаются. Выберите интерполяцию на A.')
        if not self.direct:
            self.ix,self.wx,self.valid_x = self.axis(base.x,other.x)
            self.iy,self.wy,self.valid_y = self.axis(base.y,other.y)

    @staticmethod
    def axis(target, source):
        step = source[1]-source[0]
        tolerance = step*1e-8
        valid = (target>=source[0]-tolerance)&(target<=source[-1]+tolerance)
        clipped = np.clip(target,source[0],source[-1])
        left = np.clip(np.searchsorted(source,clipped,side='right')-1,0,len(source)-2)
        weight = (clipped-source[left])/(source[left+1]-source[left])
        weight = np.where(weight<1e-8,0.,np.where(weight>1-1e-8,1.,weight))
        return left,weight,valid

    def block(self,start,end):
        if self.direct:
            return self.grid.values[start:end]
        ix,wx = self.ix[None,:],self.wx[None,:]
        iy,wy = self.iy[start:end,None],self.wy[start:end,None]
        valid = self.valid_y[start:end,None]&self.valid_x[None,:]
        result = np.zeros(valid.shape,dtype=float)
        for dy,yweight in ((0,1-wy),(1,wy)):
            for dx,xweight in ((0,1-wx),(1,wx)):
                weight = yweight*xweight
                value = self.grid.values[iy+dy,ix+dx]
                valid &= (weight==0)|np.isfinite(value)
                result += weight*np.where(np.isfinite(value),value,0.)
        return np.where(valid,result,np.nan)


_PRESET_BINARY = {
    'SUM': np.add, 'SUB': np.subtract, 'MUL': np.multiply, 'DIV': np.divide,
    'MEAN': np.add, 'MIN': np.minimum, 'MAX': np.maximum,
    'AND': lambda a,b: _logical(np.logical_and,a,b),
    'OR': lambda a,b: _logical(np.logical_or,a,b),
    'XOR': lambda a,b: _logical(np.logical_xor,a,b),
}
_PRESET_UNARY = {'NOT': lambda a: _logical(np.logical_not,a),
                 'ABS': np.abs, 'SQRT': np.sqrt, 'LOG10': np.log10}


def validate_operation(operation,count,expression=''):
    if operation not in OPERATIONS or count<1:
        raise ValueError('Выберите операцию и хотя бы одну карту.')
    if operation in UNARY_OPERATIONS and count!=1:
        raise ValueError('Эта операция требует одну карту.')
    if operation in _PRESET_BINARY and count<2:
        raise ValueError('Эта операция требует не менее двух карт.')
    if operation=='FORMULA':
        return GridFormula(expression,(operand_name(i) for i in range(count)))
    return None


def calculate_grids(grids, operation='SUM', *, expression='', alignment='strict',
                    name=None, unit=None):
    """Return a new Grid; never modify inputs. Geometry always comes from A."""
    grids = tuple(grids)
    formula = validate_operation(operation,len(grids),expression)
    if alignment not in ('strict','linear'):
        raise ValueError('Неизвестный режим согласования сеток.')
    base = grids[0]
    bindings = [(operand_name(i),grid) for i,grid in enumerate(grids)]
    used = [(key,grid) for key,grid in bindings if formula is None or key in formula.names]
    samplers = [(key,_Sampler(base,grid,alignment,key)) for key,grid in used]
    ny,nx = base.values.shape
    values = np.empty((ny,nx),dtype=float)
    # Keep temporary formula blocks bounded as the number of operands increases.
    block_rows = max(1,min(128,65536//nx,2000000//max(1,nx*len(samplers))))
    for start in range(0,ny,block_rows):
        end = min(ny,start+block_rows)
        valid = np.ones((end-start,nx),dtype=bool)
        with np.errstate(all='ignore'):
            if formula is not None:
                env = {}
                for key,sampler in samplers:
                    data = sampler.block(start,end)
                    valid &= np.isfinite(data)
                    env[key] = data
                result = formula.evaluate(env)
            else:
                first = samplers[0][1].block(start,end)
                valid &= np.isfinite(first)
                if operation in _PRESET_UNARY:
                    result = _PRESET_UNARY[operation](first)
                else:
                    result = first
                    for _,sampler in samplers[1:]:
                        data = sampler.block(start,end)
                        valid &= np.isfinite(data)
                        result = _finite(_PRESET_BINARY[operation](result,data))
                    if operation=='MEAN':
                        result = result/len(samplers)
            values[start:end] = np.where(valid & np.isfinite(result),result,np.nan)
        report(end,ny,'Калькулятор гридов')
    if not np.isfinite(values).any():
        raise ValueError('Результат не содержит конечных узлов. Проверьте формулу, пропуски и охват карт.')
    if name is None:
        name = operation+'_'+base.name
    if unit is None:
        unit = suggested_unit([grid for _,grid in used],operation)
    return Grid(base.x.copy(),base.y.copy(),values,name,unit,base.crs,base.coordinate_unit,
                base.history+[dict(operation='grid_calculator',preset=operation,
                                   expression=formula.text if formula else '',
                                   operands={key:grid.name for key,grid in used},
                                   reference=base.name,alignment=alignment)])

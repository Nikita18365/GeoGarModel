"""Memory budgets for CPU/CUDA operators, independent of the installed GPU size."""
import math

MIB = 1024**2


def available_memory(xp):
    if xp.__name__ == 'numpy':
        import psutil
        return int(psutil.virtual_memory().available)
    return int(xp.cuda.runtime.memGetInfo()[0])


def resource_plan(free_bytes, cells, observations, mode, device, policy='balanced', requested_mib=128):
    if policy not in ('balanced', 'performance', 'manual'):
        raise ValueError('Неизвестный режим памяти.')
    if not math.isfinite(requested_mib) or requested_mib < 0:
        raise ValueError('Кэш должен быть конечным и ≥0 MiB.')
    # Temporary corner integrals contain many arrays per observation/cell pair.
    # Budget conservatively; this is not a promise about total process memory.
    pair_budget=max(32768,min(1048576,int(free_bytes*.03/512)))
    ob=64 if device=='cpu' else 256
    cb=512 if device=='cpu' else max(128,2**int(math.log2(max(128,pair_budget//ob))))
    ob=min(ob,max(1,observations)); cb=min(cb,max(1,cells))
    workspace=ob*cb*512 + (cells*(3 if mode=='vector' else 1)*20+observations*16)*8
    reserve=max(256*MIB,int(free_bytes*.2))
    if workspace+reserve>=free_bytes:
        raise ValueError('Недостаточно свободной памяти для рабочих массивов и резерва. '
                         'Уменьшите модель/число наблюдений или освободите память.')
    safe=max(0,free_bytes-reserve-workspace)//MIB
    required=observations*cells*8*(1 if mode=='gravity' else 6)
    needed_mib=math.ceil(required/MIB)
    if policy=='manual':
        # A large ceiling is harmless for a small operator; only its actual size matters.
        if min(requested_mib,needed_mib)>safe:
            raise ValueError(f'Кэш требует до {min(requested_mib,needed_mib):g} MiB, '
                             f'доступный бюджет после резерва — {safe} MiB. '
                             'Уменьшите кэш или выберите автоматический режим.')
        cache=min(requested_mib,needed_mib)
    else:
        cache=min(safe,int(free_bytes*(.5 if policy=='balanced' else .7))//MIB,needed_mib)
    return dict(policy=policy,free_mib=free_bytes//MIB,cache_mib=cache,
                safe_cache_mib=safe,operator_mib=needed_mib,
                observation_block=ob,cell_block=cb)

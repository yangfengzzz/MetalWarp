def vector_add(a, b, result, n, tid):
    if tid < n:
        result[tid] = a[tid] + b[tid]

def saxpy(a, x, y, result, n, alpha, tid):
    if tid < n:
        result[tid] = alpha * x[tid] + y[tid]

def square(data, n, tid):
    if tid < n:
        data[tid] = data[tid] * data[tid]

def clamp_values(data, n, min_val, max_val, tid):
    if tid < n:
        val = data[tid]
        if val < min_val:
            data[tid] = min_val
        elif val > max_val:
            data[tid] = max_val

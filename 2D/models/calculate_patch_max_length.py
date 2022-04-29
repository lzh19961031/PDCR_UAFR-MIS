def max_value(ls,rs):
    for l in ls:
        if type(l) == list:
            rs = max_value(l,rs)
            continue
        if l > rs:
            rs = l
    return rs

def max_val(ls):
    rs = max_value(ls,0)
    return rs


def custom_round(x):
    """
    Округляет число по следующим правилам:
    - До ближайшего кратного 0.5
    - Но значения между целыми числами (например, 2.2-2.7) округляются до ближайшего .5
    - Значения ≥ X.8 округляются вверх до целого
    """
    integer_part = int(x)
    fractional_part = x - integer_part
    
    if fractional_part < 0.25:
        return integer_part
    elif 0.25 <= fractional_part < 0.75:
        return integer_part + 0.5
    else:
        return integer_part + 1
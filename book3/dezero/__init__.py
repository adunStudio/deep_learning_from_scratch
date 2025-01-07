is_simple_core = True

if is_simple_core:
    from book3.dezero.core_simple import Variable
    from book3.dezero.core_simple import Function
    from book3.dezero.core_simple import using_config
    from book3.dezero.core_simple import no_grad
    from book3.dezero.core_simple import as_array
    from book3.dezero.core_simple import as_variable
    from book3.dezero.core_simple import setup_variable

else:
    pass

setup_variable()
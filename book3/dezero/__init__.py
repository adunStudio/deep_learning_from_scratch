is_simple_core = False

if is_simple_core:
    from book3.dezero.core_simple import Variable
    from book3.dezero.core_simple import Function
    from book3.dezero.core_simple import using_config
    from book3.dezero.core_simple import no_grad
    from book3.dezero.core_simple import as_array
    from book3.dezero.core_simple import as_variable
    from book3.dezero.core_simple import setup_variable

else:
    from book3.dezero.core import Variable
    from book3.dezero.core import Function
    from book3.dezero.core import using_config
    from book3.dezero.core import no_grad
    from book3.dezero.core import as_array
    from book3.dezero.core import as_variable
    from book3.dezero.core import setup_variable

setup_variable()
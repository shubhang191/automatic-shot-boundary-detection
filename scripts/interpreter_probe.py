"""Standard library only; safe before installing dependencies."""
import json, os, struct, sys
from pathlib import Path

def check(venv=False):
    errors=[]
    if os.name!='nt' or sys.version_info[:2]!=(3,13) or struct.calcsize('P')!=8:
        errors.append('Windows with 64-bit Python 3.13 is required')
    if venv:
        expected=Path(__file__).resolve().parents[1]/'.venv'
        if Path(sys.prefix).resolve()!=expected.resolve() or sys.prefix==sys.base_prefix:
            errors.append('Interpreter is not in this application virtual environment')
        if not Path(getattr(sys,'_base_executable','')).is_file():
            errors.append('Base Python installation is missing')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,executable=sys.executable,
                prefix=sys.prefix,base_prefix=sys.base_prefix,version=sys.version,bits=struct.calcsize('P')*8)

if __name__=='__main__':
    result=check('--venv' in sys.argv)
    print(json.dumps(result));sys.exit(result['status']!='PASS')

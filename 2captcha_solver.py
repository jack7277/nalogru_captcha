# https://github.com/2captcha/2captcha-python
import _thread
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from twocaptcha import TwoCaptcha

api_key = os.getenv('APIKEY_2CAPTCHA', 'apikey2captcha')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
solver = TwoCaptcha(api_key)

# путь где лежат скачанные файлы капчи
basepath = r'c:\prj\fns-captcha\pics'
# files
captcha_files = sorted(glob.glob(f'{os.path.join(basepath, "*.jpg")}'))
captcha_files = [captcha_file for captcha_file in captcha_files if len(os.path.basename(captcha_file)) > 10]

global stop
stop = False


def solve(captcha_file):
    print(captcha_file)
    try:
        result = solver.normal(captcha_file)
        if result == 'ERROR_ZERO_BALANCE': sys.exit()
        code = result['code']
    except Exception as e:
        code = ""
        stop = True
        _thread.interrupt_main()

    if len(code) == 0:
        _thread.interrupt_main()
        sys.exit()
    else:
        new_file = f'{str(code)}.jpg'
        new_captcha_file = os.path.join(basepath, new_file)
        os.rename(captcha_file, new_captcha_file)
        print(f'solved: {str(result)}')


with ThreadPoolExecutor(max_workers=1) as executor:
    t = executor.map(solve, captcha_files)
    r = 0

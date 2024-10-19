# https://service.nalog.ru/static/captcha.bin?a=F76E355CBA309CC6374EF0ACDDAE0DB678E0A66C1A85E9441F8C2D0D58765DCB28F82A8C46ED4AEA04604718A8BB7976
import time
import requests

while True:
    r = requests.get('https://service.nalog.ru/static/captcha.bin')
    id = str(r.text)
    r = requests.get(f'https://service.nalog.ru/static/captcha.bin?a={id}&version=2')
    with open(f"pics\\{id}.jpg", "wb") as bin_file:
        bin_file.write(r.content)
    print(r.status_code, id)
    # break
    time.sleep(.251)
    if r.status_code != 200:
        time.sleep(5)
        # break
# 纯文本邮件
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import sys

from_addr = '1831948551@qq.com'
to_addr = '1831948551@qq.com'
password = 'zzeexlwsatdkchdb' # from_addr 需要授权码
# 构造邮件
def send(content):
    msg = MIMEText(content, 'plain', 'utf-8')
    msg['From'] = formataddr(('Host', from_addr)) # 随便打
    msg['To'] = formataddr(('ZHUConv', to_addr)) #随便打
    if 'error' in content:
        msg['Subject'] = 'Program Error' #邮件标题
    else:
        msg['Subject'] = 'Program Done'
    # smtp_server = 'smtp.qq.com'
    # server = smtplib.SMTP_SSL(smtp_server)
    server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # QQ邮箱的SMTP服务需SSL加密，端口为465
    # 显示发送过程
    # server.set_debuglevel(1)
    # 登陆验证 发送邮件 退出
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()
# send()

if __name__ == '__main__': 
    msg = sys.argv[1]
    send(msg)

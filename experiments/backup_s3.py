
import os

from boto.s3.key import Key
from boto.s3.connection import S3Connection

from datetime import datetime


BUCKET_NAME = 'nmtf-backup'
AWS_ACCESS_KEY_ID = 'x'
AWS_SECRET_ACCESS_KEY = 'x'
FILES_SUFFIX = '_news_results.csv'
PATH = '/home/ubuntu/project'


def get_logs():
    files = [PATH + '/' + f for f in os.listdir(PATH) if FILES_SUFFIX in f or 'out.txt' == f]
    files = [f for f in files if os.path.getsize(f) > 30]  # check if file is bigger then 30 bytes
    return files


def save(conn, files_names):
    bucket = conn.get_bucket(BUCKET_NAME)

    now = datetime.now().isoformat()

    for filename in files_names:
        key = '{}/{}'.format(now, filename)
        obj = Key(bucket)
        obj.key = key
        obj.set_contents_from_filename(filename)


def get_conn():
    conn = S3Connection(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    return conn


def main():
    conn = get_conn()
    files = get_logs()
    save(conn, files)


if __name__ == '__main__':
    main()

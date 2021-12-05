from visualdl import LogReader
import os
from visualdl import LogWriter

"""
将vdl日志文件转成普通文本文件
"""


def trans_vdlrecords_to_txt(vdl_file_path):
    txt_path = vdl_file_path.replace('.log', '.txt')
    reader = LogReader(file_path=vdl_file_path)
    tags = reader.get_tags()
    scalar = tags.get('scalar')
    if scalar is not None:
        # 打开一个文件
        fo = open(txt_path, "w")
        # print(f'tags:{tags}')
        fo.write(f'tags:{tags}')

        for tag in scalar:
            fo.write(f'\n------------{tag}------------\n')
            data = reader.get_data('scalar', tag)
            fo.write(f'The Last One is {str(data[-1])}')
            fo.write(str(data))
            # print(f'{tag}: {data}')
        # 关闭打开的文件
        fo.close()


def merge_vdlrecords(vdl_file_path, last_step_dict, writer=None):
    reader = LogReader(file_path=vdl_file_path)
    tags = reader.get_tags()
    scalar = tags.get('scalar')

    if scalar is not None:
        # print(f'tags:{tags}')
        data = None
        for tag in scalar:
            data = reader.get_data('scalar', tag)
            for e in data:
                curr_last_step = last_step_dict[tag] if tag in last_step_dict else -1
                if e.id > curr_last_step:
                    writer.add_scalar(tag=tag, step=e.id, value=e.value)
            # print(f'{tag}: {data}')
            last_step_dict[tag] = data[-1].id
            print(f'{vdl_file_path} {tag} last_id is {last_step_dict[tag]}')
    return last_step_dict

def do_merge(target_dir):
    logdir = target_dir + '/merge'
    writer = LogWriter(logdir=logdir)
    last_step_dict = {}
    for path, dir_list, file_list in os.walk(target_dir + '/logs'):
        file_list.sort()  # 保证日志顺序
        for file_name in file_list:
            if file_name.endswith('.log'):
                log_path = os.path.join(path, file_name)
                print(log_path)
                last_step = merge_vdlrecords(log_path, last_step_dict, writer)
    writer.close()

def do_trans(_dir):
    for path, dir_list, file_list in os.walk(_dir):
        for file_name in file_list:
            if file_name.endswith('.log'):
                log_path = os.path.join(path, file_name)
                print(log_path)
                trans_vdlrecords_to_txt(log_path)

if __name__ == '__main__':
    # trans_vdlrecords_to_txt('./reversal_fam_decouple/iter_3500/vdlrecords.1637903367.log')
    # trans_vdlrecords_to_txt('./sfnet-bceloss/iter_16600/vdlrecords.1638271425.log')
    # conding=utf8

    # g = os.walk(r"./reversal_fam_decouple/iter_3500")
    # g = os.walk(r"./sfnet-bceloss/iter_16600")
    # g = os.walk(r"./sfnet-origin/iter_150000")

    # try:
    #     app.run(logdir="./reversal_fam_decouple/iter_3500")
    # except Exception as err:
    #     print(err)

    # create a log file under `./log/scalar_test/train`
    # with LogWriter(logdir="./log/scalar_test/train") as writer:
    #     # use `add_scalar` to record scalar values
    #     writer.add_scalar(tag="acc", step=1, value=0.5678)
    #     writer.add_scalar(tag="acc", step=2, value=0.6878)
    #     writer.add_scalar(tag="acc", step=3, value=0.9878)
    # you can also use the following method without using context manager `with`:
    # target_dir = './sfnet-bceloss/iter_16600'
    # target_dir = './sfnet-origin/iter_150000'
    target_dir = './decoupled-segnet-origin/iter_10000'

    do_merge(target_dir)
    do_trans(target_dir + '/merge')




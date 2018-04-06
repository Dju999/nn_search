# -*- coding:utf-8 -*-

import logging

from pyhive import hive
import pickle

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)

cursor = hive.connect('ВАШ СЕРВЕР HIVE').cursor()

sql_str = """
    SELECT 
        item_id, object_id, is_serial, raw_stat.descr, 
        case when c.compilation_id is null then c.title else cmp.title end as title
    from (
                SELECT DISTINCT
                    CASE WHEN compilation_id IS NULL THEN 0 ELSE 1 END as is_serial,
                    c.id as item_id,
                    CONCAT(descr, description_alt, synopsis) as descr,
                    CASE WHEN compilation_id IS NULL THEN c.id*10 ELSE compilation_id*10+1 END as object_id
                FROM groot.content_bd1 c
                INNER JOIN groot.m2m_content_content_category m2mc
                    ON c.id = m2mc.content_id
                WHERE status = 3
                    AND m2mc.content_category_id NOT IN (1, 16, 18)
                UNION ALL
                SELECT DISTINCT
                    1 as is_serial,
                    c.id as item_id,
                    CONCAT(description, synopsis) as descr,
                    c.id * 10 + 1 as object_id
                FROM groot.compilation_bd1 c
                INNER JOIN groot.m2m_compilation_content_category_bd1 m2mc
                    ON c.id = m2mc.compilation_id
                WHERE status = 3
                    AND content_category_id NOT IN (1, 16, 18)
    ) as raw_stat
    inner join groot.content_bd1 as c
        on object_id = CASE WHEN compilation_id IS NULL THEN c.id*10 ELSE compilation_id*10+1 END
    LEFT JOIN groot.compilation_bd1 as cmp
        on cmp.id = c.compilation_id
"""


def get_content_dict():
    """Получаем словарь с контентом (популярным)

    :param MAX_BY_CATEGORY: сколько контента в каждой категории нужно собрать
    :return: 
    """
    cursor.execute(sql_str)
    data_sample = cursor.fetchall()
    logger.info('Выгрузка данных')
    col_names = [i[0] for i in cursor.description]
    data_dict = {
        i: {col_names[col_num]: data_sample[i][col_num] for col_num in range(len(col_names))}
        for i in range(len(data_sample))
    }
    final_dict = dict()
    content_key = 0
    for k in data_dict:
        entity = data_dict[k]
        final_dict[content_key] = entity
        content_key += 1
    return final_dict

content_dict = get_content_dict()

item2qid = dict()
qid2title = dict()
content_description = []
logger.info('Запись TSV')
f = open('content_descr.tsv', 'w')
for k in content_dict:
    item_id = content_dict[k]['item_id']
    qid = content_dict[k]['object_id']
    item2qid[item_id] = qid
    qid2title[int(qid)] = content_dict[k]['object_title']
    descr = content_dict[k]['descr'].replace(u'\xa0', u' ').replace('\n', ' ').replace('\t', ' ')
    f.write('{}\t{}\n'.format(item_id, descr))
f.close()
pickle.dump(item2qid, open('item2qid.pkl', 'wb'), protocol=2)
pickle.dump(qid2title, open('qid2title.pkl', 'wb'), protocol=2)

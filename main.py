from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import *

from ml_udf import label_encode

settings = EnvironmentSettings.new_instance().use_blink_planner().build()
exec_env = StreamExecutionEnvironment.get_execution_environment()
t_env = StreamTableEnvironment.create(exec_env, environment_settings=settings)

t_env.create_temporary_function("label_encode", label_encode)

CREATE_USER_TABLE_DDL = """
CREATE TABLE users (
    user_id STRING,
    source STRING,
    sex_name STRING,
    age_name STRING,
    city_name STRING,
    pic_vip_type STRING,
    lt30 STRING,
    last_pic_app_active_device_type STRING,
    last_pic_app_active_device_model STRING,
    country_name STRING,
    province_name STRING,
    is_encodephone STRING,
    is_wechat STRING
) WITH (
    'connector' = 'filesystem',
    'format' = 'csv',
    'path' = 'users.csv'
)
"""

CREATE_SINK_TABLE_DDL = """
CREATE TABLE sink (
    user_id STRING,
    a1 INT,
    a2 INT,
    a3 INT,
    a4 INT,
    a5 INT,
    a6 INT,
    a7 INT,
    a8 INT,
    a9 INT,
    a10 INT,
    a11 INT,
    a12 INT
) WITH (
    'connector' = 'print'
)
"""

TRANSFORM_DML = """
INSERT INTO sink
WITH t1 AS (
    SELECT
    user_id,
    CASE source WHEN '' THEN -1 WHEN 'NULL' THEN -1 ELSE CAST(source AS INT) END source,
    CASE sex_name WHEN '' THEN 0 WHEN 'NULL' THEN 0 ELSE CAST(sex_name AS INT) END sex_name,
    CASE age_name WHEN '' THEN -1 WHEN 'NULL' THEN -1 WHEN '100' THEN -1 ELSE CAST(age_name AS INT) END age_name,
    CASE city_name WHEN 'NULL' THEN '' ELSE city_name END city_name,
    CAST(pic_vip_type AS INT) pic_vip_type,
    CAST(lt30 AS INT) lt30,
    CAST(last_pic_app_active_device_type AS INT) last_pic_app_active_device_type,
    CASE last_pic_app_active_device_model WHEN 'NULL' THEN '' ELSE last_pic_app_active_device_model END last_pic_app_active_device_model,
    CASE country_name WHEN 'NULL' THEN '' ELSE country_name END country_name,
    CASE province_name WHEN 'NULL' THEN '' ELSE province_name END province_name,
    CAST(is_encodephone AS INT) is_encodephone,
    CAST(is_wechat AS INT) is_wechat
    FROM users
),
t2 AS (
    SELECT
    user_id,
    LABEL_ENCODE(source, 'source') source,
    LABEL_ENCODE(sex_name, 'sex_name') sex_name,
    LABEL_ENCODE(age_name, 'age_name') age_name,
    LABEL_ENCODE(city_name, 'city_name') city_name,
    LABEL_ENCODE(pic_vip_type, 'pic_vip_type') pic_vip_type,
    LABEL_ENCODE(lt30, 'lt30') lt30,
    LABEL_ENCODE(last_pic_app_active_device_type, 'last_pic_app_active_device_type') last_pic_app_active_device_type,
    LABEL_ENCODE(last_pic_app_active_device_model, 'last_pic_app_active_device_model') last_pic_app_active_device_model,
    LABEL_ENCODE(country_name, 'country_name') country_name,
    LABEL_ENCODE(province_name, 'province_name') province_name,
    LABEL_ENCODE(is_encodephone, 'is_encodephone') is_encodephone,
    LABEL_ENCODE(is_wechat, 'is_wechat') is_wechat
    FROM t1
)
SELECT
user_id,
source,
sex_name,
age_name,
city_name,
pic_vip_type,
lt30,
last_pic_app_active_device_type,
last_pic_app_active_device_model,
country_name,
province_name,
is_encodephone,
is_wechat
FROM t2
"""

t_env.execute_sql(CREATE_USER_TABLE_DDL)
t_env.execute_sql(CREATE_SINK_TABLE_DDL)
t_env.execute_sql(TRANSFORM_DML).wait()

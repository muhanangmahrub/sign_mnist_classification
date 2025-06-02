from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer
from config import db_settings

Base = declarative_base()

attrs = {
    '__tablename__': db_settings.train_table_name,
    'label': Column(Integer, primary_key=True, autoincrement=True)
}

for i in range(784):
    col_name = f'pixel{i+1}'
    attrs[col_name] = Column(Integer)

SignMnistTrain = type('SignMnistTrain', (Base,), attrs)

attrs = {
    '__tablename__': db_settings.test_table_name,
    'label': Column(Integer, primary_key=True, autoincrement=True)
}

for i in range(784):
    col_name = f'pixel{i+1}'
    attrs[col_name] = Column(Integer)

SignMnistTest = type('SignMnistTest', (Base,), attrs)

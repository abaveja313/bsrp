from sqlalchemy import Column, String, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    name = Column(String(40), nullable=False)
    password = Column(String(100), nullable=False)
    restricted = Column(Boolean)


class Query(Base):
    __tablename__ = 'query'
    id = Column(Integer, primary_key=True)
    timestamp = Column(String)
    hash = Column(String(100), nullable=False)
    file_path = Column(String(400), nullable=False)
    user_id = Column(Integer, ForeignKey('user.id'))
    scan_type = Column(Boolean)
    user = relationship(User)


engine = create_engine('sqlite:///userquery.db')
Base.metadata.create_all(engine)

from database_setup import User, Query, Base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import hashlib

engine = create_engine('sqlite:///userquery.db', connect_args={'check_same_thread': False}, echo=True)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()


def create():
    newUser = User(name='admin', password=hashlib.sha256('Amrit7391').hexdigest(), restricted=False)
    session.add(newUser)
    session.commit()


def _check(username, input_password):
    actual = session.query(User).filter_by(name=username).one()
    if hashlib.sha256(input_password).hexdigest() == actual.password:
        return True
    else:
        return False



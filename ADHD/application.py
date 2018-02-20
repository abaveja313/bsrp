from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from database_setup import Base, User, Query
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import hashlib

app = Flask(__name__)


def setup_db():
    engine = create_engine('sqlite:///userquery.db', connect_args={'check_same_thread': False}, echo=True)
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    db = DBSession()
    return db


database = setup_db()


@app.route('/')
def router():
    session['logged_in'] = False
    if session.get('logged_in'):
        return redirect('/home')
    return redirect('/login')


@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == "GET":
        if session['logged_in']:
            return redirect('/home')
        return render_template('login.html')
    else:
        try:
            user = database.query(User).filter_by(name=request.form['name']).one()
            password = hashlib.sha256(request.form['password']).hexdigest()
            if password == user.password:
                session['logged_in'] = True
                return redirect('/home')
            else:
                raise Exception
        except:
            return render_template('login.html', error='Username/Password incorrect')


if __name__ == "__main__":
    app.secret_key = 'ha_you_will_never_figure_this_out_lmao'
    app.debug = True
    app.run(host='0.0.0.0', port=5000)

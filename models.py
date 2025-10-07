from flask_sqlalchemy import SQLAlchemy

db=SQLAlchemy()


class User(db.Model):
    __tablename__="users"


    id=db.Column(db.Integer,primary_key=True)
    email=db.Column(db.String(120),unique=True,nullable=False)
    password=db.Column(db.String(128),nullable=False)

    history=db.relationship('History',backref='user',lazy=True)#let  each history row access parent user object


class History(db.Model):
    __tablename__="history"

    id=db.Column(db.Integer,primary_key=True)
    user_id=db.Column(db.Integer,db.ForeignKey('users.id'),nullable=False)
    predicted_speices=db.Column(db.String(120),nullable=False)
    confidence=db.Column(db.Float,nullable=False)
    timestamp=db.Column(db.DateTime,server_default=db.func.now())
    audio_path = db.Column(db.String(255), nullable=False)
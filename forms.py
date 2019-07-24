'''
    PROJECT STATEMENT:
        To create a ML pipeline that is able to classify headlines as
        coming from Left, Right or Center sources in the political spectrum.

    FILE STATEMENT:
        Contains the form design and characteristics
'''

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, BooleanField
from wtforms.validators import DataRequired, Length


# Create Form
class HeadlineForm(FlaskForm):
    headline = StringField('Headline', validators=[DataRequired(), Length(min=10, max=100)])
    submit = SubmitField('Get Label')

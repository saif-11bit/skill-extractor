import streamlit as st
import spacy
import re
from spacy.matcher import PhraseMatcher
from bs4 import BeautifulSoup
# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor
import numpy as np
from IPython.core.display import HTML
# init params of skill extractor
nlp = spacy.load("en_core_web_sm")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

def clean_text(raw):
    '''Case specific to be used with pandas apply method'''
    try:
        # remove carriage returns and new lines
        raw = raw.replace('\r', '')
        raw = raw.replace('\n', '')
        
        # brackets appear in all instances
        raw = raw.replace('[', '')
        raw = raw.replace(']', '')
        raw = raw.replace(')', '')
        raw = raw.replace('(', '')
        
        # removing html tags
        clean_html = re.compile('<.*?>')
        clean_text = re.sub(clean_html, ' ', raw)
        
        # removing duplicate whitespace in between words
        clean_text = re.sub(" +", " ", clean_text) 
        
        # stripping first and last white space 
        clean_text = clean_text.strip()
        
        # commas had multiple spaces before and after in each instance
        clean_text = re.sub(" , ", ", ", clean_text) 
        
        # eliminating the extra comma after a period
        clean_text = clean_text.replace('.,', '.')
        
        # using try and except due to Nan in the column
    except:
        clean_text = np.nan
        
    return clean_text

def extract_skills(job_description):
    annotations = skill_extractor.annotate(job_description)
    html_content = skill_extractor.describe(annotations)
    
    # Convert the IPython HTML object to a string
    if isinstance(html_content, HTML):
        html_string = html_content.data  # Extract the actual HTML string
    else:
        html_string = html_content
    
    # Parse the HTML and remove JavaScript attributes
    soup = BeautifulSoup(html_string, 'html.parser')
    
    # Remove all 'onmouseenter' and 'onmouseleave' attributes
    for tag in soup.find_all(True):
        if 'onmouseenter' in tag.attrs:
            del tag.attrs['onmouseenter']
        if 'onmouseleave' in tag.attrs:
            del tag.attrs['onmouseleave']
    
    clean_html = str(soup)
    return clean_html


# Streamlit app interface
def main():
    st.title("Job Description Skill Extractor")
    
    st.write("""
    This application allows you to input a job description and returns the extracted skills using the SkillNER model.
    """)
    
    # Input text box for the job description
    job_description = st.text_area("Enter Job Description", height=200)
    
    if st.button("Extract Skills"):
        if job_description:
            # Clean the input text
            cleaned_description = clean_text(job_description)
            
            # Extract skills and get the HTML output
            skills_html = extract_skills(cleaned_description)
            
            # Display the extracted skills as HTML
            st.write("### Extracted Skills")
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.warning("Please enter a job description to extract skills.")
    
if __name__ == "__main__":
    main()

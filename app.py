from  tensorflow.keras.models import load_model
model = load_model('skin_model.hdf5')

#-------
#Dermatologist API URL
import requests
derm_api_url = "https://public.opendatasoft.com/api/records/1.0/search/?dataset=medecins&q=&facet=civilite&facet=column_12&facet=column_13&facet=column_14&facet=column_16&facet=libelle_profession&facet=type_dacte_realise&facet=commune&facet=nom_epci&facet=nom_dep&facet=nom_reg&facet=insee_reg&facet=insee_dep&facet=libelle_regroupement&facet=libelle&facet=libelle_acte_clinique&facet=l&refine.commune={}&refine.libelle_profession=Dermatologue+et+vénérologue"
#-------

#Template for dermatologist search results

DERMATOLOGIST_SEARCH_RESULTS = """
<div style = width:100%;height:100%;margin:1px;padding:5px;margin:10px;position:relative;border-radius:5px;background-color:#f0f2f6>
<h5>{}<br /></h5>
<h6>{}<br /></h6>
<h7>{}<br /></h7>
<h7>{}<br /></h7>
<h8>{}<br /></h8>
</div>
"""

import streamlit as st
st.title('Bienvenue sur SkinMe !')

st.subheader("SkinMe vous permet d'établir un premier diagnosti de vos lésions de peau (type grain de beauté), d’évaluer un risque potentiel et de prendre rendez-vous chez un dermatologue si besoin.")

genre = st.radio(
"Vous êtes",
('Un homme', 'Une femme'))

age = st.text_input(label = 'Votre âge', placeholder="ex : 34, 60...")


genre = st.radio(
"Où est localisée la lésion ?",
('Cuir chevelu', 'Oreille', 'Visage', 'Dos', 'Tronc', 'Poitrine', 'Membre supérieur', 'Abdomen', 'Membre inférieur', 'Zone génitale', 'Cou', 'Main', 'Pied', 'Zone acral'))


file = st.file_uploader("Uploader la photo (un seul grain de beauté par photo) :", type=["jpg", "png","jpeg"])

#----------

import cv2
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):
    
        size = (128,128)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(128, 128),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("akiec : Actinic keratoses and intraepithelial carcinoma / Bowen's disease")
    elif np.argmax(prediction) == 1:
        st.write("bcc : basal cell carcinoma")
    elif np.argmax(prediction) == 2:
        st.write("bkl : benign keratosis-like lesions")
    elif np.argmax(prediction) == 3:
        st.write("df : dermatofibroma")
    elif np.argmax(prediction) == 4:
        st.title('Votre diagnostic:')
        st.error("Le diagnostic de votre lésion semble mauvais. Il pourrait s'agir d'un mélanome, mais pas de panique, nous vous conseillons de prendre rendez-vous chez un dermatologue afin de compléter le diagnostic par un professionnel.")
        st.write("&#128073; Ce diagnostic ne constitue pas un diagnostic médical complet, il doit être confirmé par un médecin.")
        #Find a dermatologist form
        
        st.header('Trouver un dermatologue:')
        with st.form(key="searchform"):
            first, second = st.columns([2,1])
            with first:
                city = st.text_input('Où ça ?:', placeholder="ex : '75009', '59', 'Lille', 'Creuse'...")
            with second:
                st.text(".")
                submit = st.form_submit_button("Recherche")
        #Results of the search
        col1, col2 = st.columns([2,1])
        with col1:
            if submit:
                search_url = derm_api_url.format(city)
                request = requests.get("https://public.opendatasoft.com/api/records/1.0/search/?dataset=medecins&q={}&rows=100&facet=civilite&facet=column_12&facet=column_13&facet=column_14&facet=column_16&facet=libelle_profession&facet=type_dacte_realise&facet=commune&facet=nom_epci&facet=nom_dep&facet=nom_reg&facet=insee_reg&facet=insee_dep&facet=libelle_regroupement&facet=libelle&facet=libelle_acte_clinique&facet=dep&refine.libelle_profession=Dermatologue+et+vénérologue".format(city))
                data = request.json()
                number_results = len(data['records'])
                st.subheader("{} dermatologues trouvés:".format(number_results, city))

                for i in range(0,number_results):
                    name = data['records'][i]['fields']['nom']
                    #st.write(name)
                    job = data['records'][i]['fields']['libelle_profession']
                    #st.write(job)
                    adress = data['records'][i]['fields']['adresse']
                    #st.write(adress)
                    try:
                        tel = data['records'][i]['fields']['column_10']
                    except KeyError:
                        print (" ")
                    #st.write(tel)
                    conv = data['records'][i]['fields']['column_14']
                    st.markdown(DERMATOLOGIST_SEARCH_RESULTS.format(name,job,adress,tel,conv),unsafe_allow_html=True)
    


    elif np.argmax(prediction) == 5:
        st.title('Votre diagnostic:')
        st.success("Tout va bien, il s'agit d'un nævus mélanocytaire, autrement dit, un simple grain de beauté !")
        st.write("Ce diagnostic ne constitue pas un diagnostic médical complet, il doit être confirmé par un médecin.")
           
    else:
        st.title('Votre diagnostic:')
        st.warning("Votre lésion semble appartenir à la famille des lésions vasculaires, famille très large, dont seul un médecin peut dresser le diagnostic précis. Pas d'urgence, mais n'hésitez pas à prévoir un simple rendez-vous de contôle chez un dermatologue pour être rassuré(e).")
        st.write("Ce diagnostic ne constitue pas un diagnostic médical complet, il doit être confirmé par un médecin.")
        st.header('Trouver un dermatologue:')
        choix = st.selectbox(
        "Votre département",
        ("01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45","46","47","48","49","50","51","52","53","54","55","56","57","58","59","60","61","62","63","64","65","66","67","68","69","70","71","72","73","74","75","76","77","78","79","80","81","82","83","84","85","86","87","88","89","90","91","92","93","94","95","971","972","973","974","976"))

    
    
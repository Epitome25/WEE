import streamlit as st
import cv2 as cv
import numpy as np
import keras
import requests
from geopy.geocoders import Nominatim

# Define the labels for the diseases
label_name = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
    'Cherry Powdery mildew', 'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot',
    'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black rot',
    'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot',
    'Peach healthy', 'Pepper bell Bacterial spot', 'Pepper bell healthy',
    'Potato Early blight', 'Potato Late blight', 'Potato healthy',
    'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot',
    'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold',
    'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# Instructions for the user
st.write("Please input only leaf Images of Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato. Otherwise, the model will not work perfectly.")

# Load the model with error handling
try:
    model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Define disease treatments
disease_treatments = {'Tomato Bacterial Spot':{'Pesticides-Actigard 50WGb alternated with copper + Mancozeb.Management Practices-Avoid sprinkler irrigation, maintain proper distance from cull piles near greenhouses or fields, and implement crop rotation with nonhost crops. Apply bactericides at intervals of 7 to 10 days, with shorter intervals recommended in conditions of rain, high humidity, and warm temperatures.'},

'Tomato Early Blight':{'Fungicide-GardenTech brand Daconil fungicides.Cultural Practices- Apply mulch, such as fabric, straw, plastic, or dried leaves, to cover the soil around plants. Employ drip irrigation to water at the base of the plants, and remove infected leaves promptly, ensuring they are either buried or burned to prevent further spread.'},
 
'Tomato late blight' :{'Fungicide - Ridomil Gold (Metalaxyl 4% + Mancozeb 64%).Management Practices: Preventative fungicide sprays may be warranted if late blight is present in the surrounding area. Infected plants should be removed, destroyed, and properly disposed of to curb disease transmission.'},

 'Tomato Leaf Mold': {'Fungicide: GSC Organic Tomato Fertilizer.Cultural Practices: Maintain relative humidity below 85PERCENTAGE and ensure night temperatures exceed outdoor temperatures, particularly in greenhouses. Utilize drip irrigation, avoid wetting the foliage, and prune or stake the plants to promote upright growth and improve airflow around the plants.'}, 

'Tomato Septoria Leaf Spot': {'Fungicide: Organocide Plant Doctor Systemic Fungicide. Cultural Practices: Remove diseased leaves and improve air circulation around the plants. Apply mulch around the base of the plants, avoid overhead watering, and control weed growth. Crop rotation is also recommended to reduce the incidence of the disease.'}, 

'Tomato Spider Mites':{ 'Treatment: Prepare a chilli solution by mixing 20g of pounded chilli with 1 liter of water, allowing it to sit for one day, then dilute with 5 liters of water. Apply the solution weekly to control newly hatched mites. Alternatively, use a hard stream of water to dislodge the mites or apply insecticidal soaps, horticultural oils, or neem oil.'}, 

'Tomato Target Spot': {'Fungicides: Chlorothalonil, copper oxychloride, or mancozeb. Management Practices: Begin treatment at the first sign of lesions and continue applications at intervals of 10 to 14 days, ceasing 3 to 4 weeks before the final harvest.', 'Tomato Yellow Leaf Curl Virus:' 'Management Practices: Infected plants should be covered with a clear or black plastic bag tied at the stem. Cut the plant below the bag and allow it to desiccate for 1 to 2 days before disposal. No treatments exist for the virus itself; therefore, controlling whitefly populations is crucial to prevent the spread of infection.'},

'Tomato Mosaic Virus': {'Management Practices: Promptly remove infected plants, as they act as a source of infection for healthy plants nearby. After handling infected plants, wash hands and tools with hot, soapy water. Sterilize tools using a disinfectant such as Virkon S to minimize cross-contamination. Additionally, avoid planting other susceptible species in proximity to infected plants.'},

'Blackrot': {'Black rot thrives in humid climates, with berries remaining highly susceptible for 3-5 weeks after cap fall and becoming immune after 2 more weeks. The primary infection source is mummified berries from the previous season, making vineyard sanitation crucial for prevention. The disease spreads from leaves to fruit, potentially causing complete crop loss in severe cases. Effective control includes fungicides like mancozeb and ziram, while DMIs and strobilurins (e.g., Abound, Aprovia Top, Pristine, Quadris Top) provide strong protection.'},
 
'grape esca':{'Esca disease is caused by fungi like Phaeomoniella chlamydospora and Phaeoacremonium minimum, which enter through wounds and spread via the vines vascular system.Common symptoms include yellowing or reddening of leaves, necrotic shoots, and cankers on vine wood. Early detection is crucial to prevent severe damage.Sterilizing pruning tools with disinfectants helps reduce the spread of infection between plants.Fungicide and bactericide treatments can help control the disease, and selecting resistant grape varieties offers long-term protection.Maintaining well-drained soil and proper root management prevents water stagnation and enhances vine health.'},

'grape leaf blight':{'Bacterial blight of grapevine, caused by Xanthomonas ampelina, spreads through pruning tools and enters healthy tissues via wounds, especially in wet conditions. Early symptoms include reddish-brown streaks, cankers, and shoot dieback. Severely infected vines appear stunted, with dead leaves and blackened flowers. Infected roots result in poor shoot growth, affecting vine health.The disease can cause serious harvest losses, requiring laboratory confirmation for accurate diagnosis.Preventive measures include sterilizing tools, sourcing healthy plants, regular vineyard monitoring, and enforcing hygiene protocols.Following biosecurity practices, such as “Come clean, Go clean,” helps prevent the spread of bacterial blight.'},

 'Peach Bacterial Spot':{ 'Peach bacterial spot caused by Xanthomonas arboricola pv. pruni, affects leaves, fruit, and twigs, leading to dark water-soaked lesions, yellowing, and premature leaf drop. Severe infections reduce fruit quality and yield.The disease spreads through wind, rain, and infected plant material, thriving in warm, humid conditions. Symptoms include small dark spots on fruit that enlarge and crack, causing blemishes.To prevent bacterial spot, use resistant peach varieties and certified disease-free plants. Maintain proper pruning and airflow to reduce moisture buildup.Apply copper-based bactericides or antibiotics as preventive measures during the growing season. Avoid excessive nitrogen fertilization, which makes trees more vulnerable.Follow biosecurity measures like cleaning tools, removing infected plant material, and regular orchard monitoring to prevent the spread of the disease.'},

'Pepper Bell Bacterial Spot': {'Cause: Xanthomonas campestris pv. vesicatoria spreads through contaminated seeds, plant debris, wind, and rain.Symptoms: Dark, water-soaked lesions on leaves, stems, and fruits, leading to defoliation and reduced yield.Prevention: Use disease-free seeds, apply copper-based bactericides, ensure good air circulation, and avoid overhead watering.'},

'Potato Early Blight':{'Cause: Alternaria solani, a fungal pathogen, thrives in warm, humid conditions and spreads via wind, water, and infected debris. Symptoms: Brown concentric-ring lesions on leaves, progressing to yellowing and defoliation, reducing tuber size. Prevention: Rotate crops, apply fungicides, remove infected plant debris, and ensure proper plant nutrition to enhance resistance.'},

'Potato Late Blight':{'Cause: Phytophthora infestans, a water mold, spreads via rain, wind, and contaminated soil or tools, Symptoms: Dark, water-soaked spots on leaves, white mold under humid conditions, and rapid plant decay.Prevention: Use resistant varieties, apply fungicides, avoid excess moisture, and remove infected plants immediately.'},

'Strawberry Leaf Scorch':{'Cause: Diplocarpon earlianum, a fungal pathogen, spreads in wet, humid conditions through rain splash and infected debris.Symptoms: Purple-brown leaf spots that expand, causing scorched, dried leaves, weakening plant vigor.Prevention: Improve air circulation, prune excess growth, apply fungicides, and remove infected leaves to prevent spread.'},

'Apple Scab':{'Cause:Apple scab is caused by the fungal pathogen Venturia inaequalis. It thrives in wet and humid conditions, leading to dark, scaly lesions on leaves and fruit. Treatment:Use fungicides such as fixed copper, Bordeaux mixtures, copper soaps (copper octanoate), sulfur, mineral or neem oils, and myclobutanil.Myclobutanil is a synthetic fungicide, while the others are considered organically acceptable.Apply copper- or sulfur-based protectant sprays to prevent the disease.'},

'Apple Black Rot': {'Cause:Apple black rot is caused by the fungus Diplodia seriata. The disease is promoted by warm temperatures, frequent rain, and prolonged wet conditions, which facilitate fruit infection.Treatment:Apply copper-based products, lime-sulfur, Daconil, Ziram, or Mancozeb as fungicidal treatments.Remove mummified fruit (shriveled and dried fruit) attached to the tree to prevent further spread.'},

'Cedar Apple Rust': {'Cause:Cedar apple rust is caused by multiple fungal species from the genus Gymnosporangium. This disease requires both apple and juniper trees for its life cycle, making proximity a key factor in its spread. Treatment: Use fungicides containing myclobutanil, which are the most effective in preventing rust.Copper and sulfur-based products can also be applied.Maintain a minimum distance of one mile between apple and juniper trees to reduce infection risk.'},

'Cherry Powdery Mildew':{'Cause:Cherry powdery mildew is caused by the fungus Podosphaera clandestina. It thrives in warm, humid conditions with low rainfall. Treatment:Apply fungicides soon after petal fall and repeat 2 to 3 weeks later if necessary.Recommended fungicides include:Bonide Sulfur Plant Fungicide, Hi-Yield Snake Eyes Dusting Wettable SulfurMonterey Horticultural Oil,Spectracide IMMUNOX Multi-Purpose Fungicide Spray Concentrate,Tebucon 45 DF, Tesaris, Topguard SC, Topguard EQ, Topsin 4.5 FL, Torino'},

'Cercospora Leaf Spot':{'Cause:Cercospora leaf spot is caused by multiple species of Cercospora fungi. The disease spreads via airborne conidia (fungal spores transported through the air).Treatment:Apply fungicides, including:Mancozeb (400 g/acre),Copper oxychloride (500 g/acre),Carbendazim (200 g/acre),Propiconazole (200 ml/acre),Metiram (200 g/acre),Kresoxim-methyl (200 ml/acre),SYSTHANE (contains myclobutanil),HERITAGE (contains azoxystrobin),Remove infected plant material to prevent further spread.'},

'Corn Common Rust':{'Cause:Common rust in corn is caused by the fungus Puccinia sorghi.Treatment:Use fungicides such as:Chlorothalonil,Mancozeb,Kresoxim-methyl,Tebuconazole,Pyraclostrobin,Azoxystrobin,Plant corn hybrids with genetic resistance to the disease.'},

'Corn Northern Leaf Blight':{ 'Cause: Northern leaf blight in corn is caused by the fungus Exserohilum turcicum.Treatment:Apply appropriate fungicides to control the infection.Use endophytic Trichoderma as a biological control agent to suppress the disease. CREATE  A DICTIONARY WHERE THE DIESASE NAMES ARE KEYS AND THE TREATMENT ,CAUSE,PESTICIDES ALL COME UNDER VALUE'},

'Tomato Bacterial Spot' :{'Pesticides-Actigard 50WGb alternated with copper + Mancozeb.Management Practices-Avoid sprinkler irrigation, maintain proper distance from cull piles near greenhouses or fields, and implement crop rotation with nonhost crops. Apply bactericides at intervals of 7 to 10 days, with shorter intervals recommended in conditions of rain, high humidity, and warm temperatures.'},

'Tomato Early Blight':{'Fungicide-GardenTech brand Daconil fungicides.Cultural Practices- Apply mulch, such as fabric, straw, plastic, or dried leaves, to cover the soil around plants. Employ drip irrigation to water at the base of the plants, and remove infected leaves promptly, ensuring they are either buried or burned to prevent further spread.'},

'Tomato late blight':{' Fungicide - Ridomil Gold (Metalaxyl 4% + Mancozeb 64%).Management Practices: Preventative fungicide sprays may be warranted if late blight is present in the surrounding area. Infected plants should be removed, destroyed, and properly disposed of to curb disease transmission.'},

'Tomato Leaf Mold':{'Fungicide: GSC Organic Tomato Fertilizer.Cultural Practices: Maintain relative humidity below 85% and ensure night temperatures exceed outdoor temperatures, particularly in greenhouses. Utilize drip irrigation, avoid wetting the foliage, and prune or stake the plants to promote upright growth and improve airflow around the plants.'},

'Tomato Septoria Leaf Spot':{'Fungicide: Organocide Plant Doctor Systemic Fungicide.Cultural Practices: Remove diseased leaves and improve air circulation around the plants. Apply mulch around the base of the plants, avoid overhead watering, and control weed growth. Crop rotation is also recommended to reduce the incidence of the disease.'},

'Tomato Spider Mites' :{'Treatment: Prepare a chilli solution by mixing 20g of pounded chilli with 1 liter of water, allowing it to sit for one day, then dilute with 5 liters of water. Apply the solution weekly to control newly hatched mites. Alternatively, use a hard stream of water to dislodge the mites or apply insecticidal soaps, horticultural oils, or neem oil.'},

'Tomato Target Spot':{'Fungicides: Chlorothalonil, copper oxychloride, or mancozeb.Management Practices: Begin treatment at the first sign of lesions and continue applications at intervals of 10 to 14 days, ceasing 3 to 4 weeks before the final harvest.'},

'Tomato Yellow Leaf Curl Virus':{'Management Practices: Infected plants should be covered with a clear or black plastic bag tied at the stem. Cut the plant below the bag and allow it to desiccate for 1 to 2 days before disposal. No treatments exist for the virus itself; therefore, controlling whitefly populations is crucial to prevent the spread of infection.'},

'Tomato Mosaic Virus': {'Management Practices: Promptly remove infected plants, as they act as a source of infection for healthy plants nearby. After handling infected plants, wash hands and tools with hot, soapy water. Sterilize tools using a disinfectant such as Virkon S to minimize cross-contamination. Additionally, avoid planting other susceptible species in proximity to infected plants.'},

'Blackrot':{'Black rot thrives in humid climates, with berries remaining highly susceptible for 3-5 weeks after cap fall and becoming immune after 2 more weeks. The primary infection source is mummified berries from the previous season, making vineyard sanitation crucial for prevention. The disease spreads from leaves to fruit, potentially causing complete crop loss in severe cases. Effective control includes fungicides like mancozeb and ziram, while DMIs and strobilurins (e.g., Abound, Aprovia Top, Pristine, Quadris Top) provide strong protection.'},

'Grape Esca':{'Esca disease is caused by fungi like Phaeomoniella chlamydospora and Phaeoacremonium minimum, which enter through wounds and spread via the vines vascular system.Common symptoms include yellowing or reddening of leaves, necrotic shoots, and cankers on vine wood. Early detection is crucial to prevent severe damage.Sterilizing pruning tools with disinfectants helps reduce the spread of infection between plants.Fungicide and bactericide treatments can help control the disease, and selecting resistant grape varieties offers long-term protection.Maintaining well-drained soil and proper root management prevents water stagnation and enhances vine health.'},

'Grape Leaf Blight':{'Bacterial blight of grapevine, caused by Xanthomonas ampelina, spreads through pruning tools and enters healthy tissues via wounds, especially in wet conditions. Early symptoms include reddish-brown streaks, cankers, and shoot dieback.Severely infected vines appear stunted, with dead leaves and blackened flowers. Infected roots result in poor shoot growth, affecting vine health.The disease can cause serious harvest losses, requiring laboratory confirmation for accurate diagnosis.Preventive measures include sterilizing tools, sourcing healthy plants, regular vineyard monitoring, and enforcing hygiene protocols.Following biosecurity practices, such as “Come clean, Go clean,” helps prevent the spread of bacterial blight.'},

'Peach Bacteria':{'Peach bacterial spot caused by Xanthomonas arboricola pv. pruni, affects leaves, fruit, and twigs, leading to dark water-soaked lesions, yellowing, and premature leaf drop. Severe infections reduce fruit quality and yield.The disease spreads through wind, rain, and infected plant material, thriving in warm, humid conditions. Symptoms include small dark spots on fruit that enlarge and crack, causing blemishes.To prevent bacterial spot, use resistant peach varieties and certified disease-free plants. Maintain proper pruning and airflow to reduce moisture buildup.Apply copper-based bactericides or antibiotics as preventive measures during the growing season. Avoid excessive nitrogen fertilization, which makes trees more vulnerable. Follow biosecurity measures like cleaning tools, removing infected plant material, and regular orchard monitoring to prevent the spread of the disease.'},

'Pepper Bell Bacterial Spot':{'Cause: Xanthomonas campestris pv. vesicatoria spreads through contaminated seeds, plant debris, wind, and rain.Symptoms: Dark, water-soaked lesions on leaves, stems, and fruits, leading to defoliation and reduced yield.Prevention: Use disease-free seeds, apply copper-based bactericides, ensure good air circulation, and avoid overhead watering.'},

'Potato Early Blight':{'Cause: Alternaria solani, a fungal pathogen, thrives in warm, humid conditions and spreads via wind, water, and infected debris.Symptoms: Brown concentric-ring lesions on leaves, progressing to yellowing and defoliation, reducing tuber size.Prevention: Rotate crops, apply fungicides, remove infected plant debris, and ensure proper plant nutrition to enhance resistance.'},

'Potato Late Blight':{'Cause: Phytophthora infestans, a water mold, spreads via rain, wind, and contaminated soil or tools Symptoms: Dark, water-soaked spots on leaves, white mold under humid conditions, and rapid plant decay.Prevention: Use resistant varieties, apply fungicides, avoid excess moisture, and remove infected plants immediately.'},

'Strawberry Leaf Scorch':{'Cause: Diplocarpon earlianum, a fungal pathogen, spreads in wet, humid conditions through rain splash and infected debris.Symptoms: Purple-brown leaf spots that expand, causing scorched, dried leaves, weakening plant vigor.Prevention: Improve air circulation, prune excess growth, apply fungicides, and remove infected leaves to prevent spread.'},

'Apple Scab':{'Cause:Apple scab is caused by the fungal pathogen Venturia inaequalis. It thrives in wet and humid conditions, leading to dark, scaly lesions on leaves and fruit.Treatment :Use fungicides such as fixed copper, Bordeaux mixtures, copper soaps (copper octanoate), sulfur, mineral or neem oils, and myclobutanil.Myclobutanil is a synthetic fungicide, while the others are considered organically acceptable.Apply copper- or sulfur-based protectant sprays to prevent the disease.'},

'Apple Black Rot':{'Cause:Apple black rot is caused by the fungus Diplodia seriata. The disease is promoted by warm temperatures, frequent rain, and prolonged wet conditions, which facilitate fruit infection. Treatment:Apply copper-based products, lime-sulfur, Daconil, Ziram, or Mancozeb as fungicidal treatments.Remove mummified fruit (shriveled and dried fruit) attached to the tree to prevent further spread.'},

'Cedar Apple Rust':{'Cause:Cedar apple rust is caused by multiple fungal species from the genus Gymnosporangium. This disease requires both apple and juniper trees for its life cycle, making proximity a key factor in its spread.Treatment:Use fungicides containing myclobutanil, which are the most effective in preventing rust.Copper and sulfur-based products can also be applied.Maintain a minimum distance of one mile between apple and juniper trees to reduce infection risk.'},

'Cherry Powdery Mildew':{'Cause:Cherry powdery mildew is caused by the fungus Podosphaera clandestina. It thrives in warm, humid conditions with low rainfall.Treatment:Apply fungicides soon after petal fall and repeat 2 to 3 weeks later if necessary.Recommended fungicides include:Bonide Sulfur Plant Fungicide,Hi-Yield Snake Eyes Dusting Wettable Sulfur,Monterey Horticultural Oil,Spectracide IMMUNOX Multi-Purpose Fungicide Spray Concentrate,Tebucon 45 DF, Tesaris, Topguard SC, Topguard EQ, Topsin 4.5 FL, Torino.'},

'Cercospora Leaf Spot':{'Cause:Cercospora leaf spot is caused by multiple species of Cercospora fungi. The disease spreads via airborne conidia (fungal spores transported through the air).Treatment:Apply fungicides, including:Mancozeb (400 g/acre),Copper oxychloride (500 g/acre),Carbendazim (200 g/acre),Propiconazole (200 ml/acre),Metiram (200 g/acre),Kresoxim-methyl (200 ml/acre),SYSTHANE (contains myclobutanil),HERITAGE (contains azoxystrobin).Remove infected plant material to prevent further spread.'},

'Corn Common Rust':{'Cause:Common rust in corn is caused by the fungus Puccinia sorghi.Treatment:Use fungicides such as:Chlorothalonil,Mancozeb,Kresoxim-methyl,Tebuconazole,Pyraclostrobin,Azoxystrobin,Plant corn hybrids with genetic resistance to the disease.'},

'Corn Northern Leaf Blight':{'Cause:Northern leaf blight in corn is caused by the fungus Exserohilum turcicum.Treatment:Apply appropriate fungicides to control the infection.Use endophytic Trichoderma as a biological control agent to suppress the disease.'}}


# File uploader for image input
uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
    predictions = model.predict(normalized_image)
    st.image(image_bytes)

    if predictions[0][np.argmax(predictions)] * 100 >= 80:
        st.write(f"Result is: {label_name[np.argmax(predictions)]}")
        disease_name = label_name[np.argmax(predictions)]
        formatted_disease_name = disease_name.title()  # Converts to Title Case
        if formatted_disease_name in disease_treatments:
            st.write(disease_treatments[formatted_disease_name])
        else:
            st.write("No treatment found for this disease.")
    else:
        st.write("Try Another Image")

# Function to find nearest fertilizer shops
def find_nearest_fertilizer_shops(location):
    geolocator = Nominatim(user_agent="fertilizer-finder", timeout=10)

    try:
        geocode = geolocator.geocode(location)
        if not geocode:
            return None, "❌ Location not found. Please enter a valid location."

        latitude, longitude = geocode.latitude, geocode.longitude
        st.session_state["latitude"] = latitude
        st.session_state["longitude"] = longitude

        # Overpass API for querying nearby fertilizer shops
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
                        [out:json];
                        (
                            node["shop"="agrarian"](around:5000,{latitude},{longitude});
                            node["shop"="farm"](around:5000,{latitude},{longitude});
                        );
                        out body;
                        """

        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if 'elements' in data and data['elements']:
                results = [
                    {
                        "name": elem.get("tags", {}).get("name", "Unknown"),
                        "latitude": elem["lat"],
                        "longitude": elem["lon"],
                        "address": elem.get("tags", {}).get("addr:full", "Address not available"),
                    }
                    for elem in data['elements']
                ]
                return results, None
            else:
                return None, "⚠️ No nearby fertilizer shops found."
        else:
            return None, f"🚨 Overpass API error: {response.status_code}"

    except Exception as e:
        return None, f"⚠️ Error occurred: {str(e)}"

# Streamlit UI for finding fertilizer shops
st.title("🌱 Nearest Fertilizer Shops Finder")

# Store user input in session state
if "location" not in st.session_state:
    st.session_state["location"] = ""

# Allow user to enter location
st.session_state["location"] = st.text_input("📍 Enter your location:")

# Button to trigger search
if st.button("Find Fertilizer Shops Nearby"):
    if st.session_state["location"]:
        fertilizer_shops, error = find_nearest_fertilizer_shops(st.session_state["location"])

        # Display location details
        if "latitude" in st.session_state and "longitude" in st.session_state:
            st.write(f"✅ **Location Found:** {st.session_state['location']}")
            st.write(f"🌍 **Latitude:** {st.session_state['latitude']}, **Longitude:** {st.session_state['longitude']}")

        if error:
            st.error(error)
        else:
            st.write("### 🏪 Nearby Fertilizer Shops:")
            for i, shop in enumerate(fertilizer_shops, 1):
                st.write(f"**{i}. {shop['name']}**")
                st.write(f"📍 {shop['address']}")
                st.write(f"🌍 Latitude: {shop['latitude']}, Longitude: {shop['longitude']}")
                st.write("---")
    else:
        st.warning("⚠️ Please enter a valid location.")

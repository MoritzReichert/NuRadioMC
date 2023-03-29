import time
import streamlit as st
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page
from NuRadioReco.detector.webinterface.utils.helper_station import insert_station_position_to_db, load_general_station_infos, load_measurement_station_position_information, load_measurement_names
from datetime import datetime
from datetime import time

page_name = 'station_position'


def validate_inputs(cont, measurement_name, station_id, measurement_time):
    disable_insert_in_db = True

    check_measurement_name = False
    check_station_in_db = False
    check_measurement_time = False
    check_measurement_name_in_db = False

    # check if a measurement name as a input is given
    if measurement_name != '' and measurement_name != 'new measurement':
        check_measurement_name = True
    else:
        cont.error('No measurement name is entered.')

    # check that there is a entry for the selected station in the database
    general_station_info = load_general_station_infos(station_id, 'station_rnog')
    if general_station_info != {}:
        check_station_in_db = True
    else:
        cont.error('There is no corresponding station in the database. Please insert general station information first.')

    # check that there is no existing entry for this measurement in the database
    measurement_info_db = load_measurement_station_position_information(station_id, measurement_name)
    if measurement_name != '':
        if measurement_info_db == [] or measurement_info_db == {}:
            check_measurement_name_in_db = True
        else:
            cont.error('The measurement is already inserted in the database. Please choose a different measurement name.')

    # check that the measurement time is within the commission time of the station
    if general_station_info[station_id]['commission_time'] < measurement_time < general_station_info[station_id]['decommission_time']:
        check_measurement_time = True
    else:
        cont.error('The measurement time is not within the commission and decommission time of the station.')

    if check_measurement_time and check_measurement_name_in_db and check_measurement_name and check_station_in_db:
        disable_insert_in_db = False

    return disable_insert_in_db


def build_main_page(main_cont):
    # is primary measurement?
    primary = main_cont.checkbox('Is this the primary measurement?', value=True)

    # measurement time
    measurement_time = main_cont.date_input('Enter the time of the measurement:', value=datetime.utcnow(), help='The date when the measurement was conducted.')
    measurement_time = datetime.combine(measurement_time, time(0, 0, 0))

    # position
    col1_pos, col2_pos, col3_pos = main_cont.columns([1, 1, 1])
    default_position = [0, 0, 0]
    x_pos = col1_pos.text_input('x position', value=default_position[0])
    y_pos = col2_pos.text_input('y position', value=default_position[1])
    z_pos = col3_pos.text_input('z position', value=default_position[2])
    position = [float(x_pos), float(y_pos), float(z_pos)]

    cont_station_warning = main_cont.container()
    disable_station_button = validate_inputs(cont_station_warning, measurement_name, selected_station_id, measurement_time)

    insert_station = main_cont.button('INSERT STATION TO DB', disabled=disable_station_button)

    cont_station_info = main_cont.container()

    if insert_station:
        insert_station_position_to_db(selected_station_id, measurement_name, measurement_time, position, primary)
        st.session_state.insert_station = True
        st.session_state.key = '1'
        st.experimental_rerun()


# main page setup
page_configuration()

if 'station_success' not in st.session_state:
    st.session_state['station_success'] = False

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

st.title('Add station position information')
st.markdown(page_name)

# enter the information for the single stations
cont_warning_top = st.container()
station_list = ['Station 11 (Nanoq)', 'Station 12 (Terianniaq)', 'Station 13 (Ukaleq)', 'Station 14 (Tuttu)', 'Station 15 (Umimmak)', 'Station 21 (Amaroq)', 'Station 22 (Avinngaq)', 'Station 23 (Ukaliatsiaq)',
                'Station 24 (Qappik)', 'Station 25 (Aataaq)']
selected_station = st.selectbox('Select a station', station_list)
# get the name and id out of the string
selected_station_id = int(selected_station[len('Station '):len('Station ') + 2])

# select a unique name for the measurement (survey_01, tape_measurement, ...)
col1_name, col2_name = st.columns([1, 1])
measurement_list = load_measurement_names('station_position')
measurement_list.insert(0, 'new measurement')
selected_name = col1_name.selectbox('Select a measurement or enter a unique name:', measurement_list)
text_input_disabled = True
if selected_name == 'new measurement':
    text_input_disabled = False
name_input = col2_name.text_input('Select a measurement or enter a unique name:', label_visibility='hidden', disabled=text_input_disabled)
measurement_name = selected_name
if measurement_name == 'new measurement':
    measurement_name = name_input

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun

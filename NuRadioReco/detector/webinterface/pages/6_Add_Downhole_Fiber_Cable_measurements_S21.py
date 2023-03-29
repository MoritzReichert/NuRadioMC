import streamlit as st
from NuRadioReco.detector.webinterface.utils.page_config import page_configuration
from NuRadioReco.detector.webinterface.utils.helper import build_success_page, single_S_data_validation, create_double_plot
from NuRadioReco.detector.webinterface.utils.helper_cable import select_cable, validate_global_cable, insert_cable_to_db
from NuRadioReco.detector.webinterface.utils.helper_protocol import load_measurement_protocols_from_db

page_name = 'downhole_cable'
s_name = 'S21'


def build_main_page(main_cont):
    main_cont.title('Add S21 measurements for a CABLE')
    main_cont.markdown(page_name)
    cont_warning_top = main_cont.container()

    # select the cable
    cable_name = select_cable(page_name, main_cont, cont_warning_top)

    # input of checkboxes, data_format, units and protocol
    working = main_cont.checkbox('channel is working', value=True)
    primary = main_cont.checkbox('Is this the primary measurement?', value=True)
    data_stat = main_cont.selectbox('specify the data format:', ['comma separated \",\"'])
    input_units = ['', '', '']
    col11, col22, col33 = main_cont.columns([1, 1, 1])
    input_units[0] = col11.selectbox('Units:', ['Hz', 'GHz', 'MHz'])
    input_units[1] = col22.selectbox('', ['dB', 'MAG'])
    input_units[2] = col33.selectbox('', ['deg', 'rad'])
    protocols_db = load_measurement_protocols_from_db()
    protocol = main_cont.selectbox('Specify the measurement protocol: (description of the protocols can be found [here](https://radio.uchicago.edu/wiki/index.php/Measurement_protocols))', protocols_db,
                                   help='Your measurement protocol is not listed? Please add it to the database [here](Add_measurement_protocol)')

    # upload the data
    uploaded_magnitude = main_cont.file_uploader('Select your magnitude measurement:', accept_multiple_files=False, key=page_name + 'mag')
    uploaded_phase = main_cont.file_uploader('Select your phase measurement:', accept_multiple_files=False, key=page_name + 'phase')

    # container for warnings/infos at the botton
    cont_warning_bottom = main_cont.container()

    # # input checks and enable the INSERT button
    S_magnitude, smagnitude_validated = single_S_data_validation(cont_warning_bottom, uploaded_magnitude, [input_units[0], input_units[1]], 'magnitude')
    S_phase, sphase_validated = single_S_data_validation(cont_warning_bottom, uploaded_phase, [input_units[0], input_units[2]], 'phase')
    disable_button = validate_global_cable(cont_warning_bottom, cable_name, working, smagnitude_validated, sphase_validated, uploaded_magnitude, uploaded_phase)
    figure = create_double_plot(s_name, cont_warning_bottom, S_magnitude, S_phase, ['frequency', 'magnitude'], ['frequency', 'phase'], ['MHz', input_units[1]], ['MHz', input_units[2]])
    main_cont.plotly_chart(figure, use_container_width=True)

    # INSERT button
    upload_button = main_cont.button('INSERT TO DB', disabled=disable_button)

    # insert the data into the database and change to the success page by setting the session key
    if upload_button:
        insert_cable_to_db(page_name, s_name, cable_name, S_magnitude, S_phase, input_units, working, primary, protocol)
        main_cont.empty()
        st.session_state.key = '1'
        st.experimental_rerun()


# main page setup
page_configuration()

# initialize the session key (will be used to display different pages depending on the button clicked)
if 'key' not in st.session_state:
    st.session_state['key'] = '0'

main_container = st.container()  # container for the main part of the page (with all the input filed)
success_container = st.container()  # container to display the page when the data was submitted

if st.session_state.key == '0':
    build_main_page(main_container)  # after clicking the submit button, the session key is set to '1' and the page is rerun

if st.session_state.key == '1':
    build_success_page(success_container, page_name)  # after clicking the 'add another measurement' button, the session key is set to '0' and the page is rerun

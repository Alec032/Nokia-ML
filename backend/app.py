from json_operations import get_documents
from user_input import process_input
from feature_extraction import extract_features, extract_other
from tfidf_operations import prepare
from scipy.sparse import load_npz
from model_load import load_model
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from load_encoder import load_encoder

db_name2 = 'encoder_db'
collection_name2 = 'label_encoders'
encoder_name = 'group_in_charge_encoder'

client = 'mongodb://localhost:27017/'
db_name = 'model_db'
collection_name = 'MultinomialNB_model_final2'
model_name = 'MultinomialNB'

def predict(data):
    choice = data.get('choice')

    if choice == '1':
        title = data.get('title')
        description = data.get('description')
        build = data.get('build')
        feature = data.get('feature')
        release = data.get('release')

        combined_feature_tokens, other_feature_tokens = process_input(title, description, build, feature, release)
        if not combined_feature_tokens or not other_feature_tokens:
            return {'error': 'Failed to process input'}

        output_file = "output.npz"
        prepare(combined_feature_tokens, other_feature_tokens, output_file)

    elif choice == '2':
        json_file_path = os.path.normpath(data.get('json_file_path'))
        if not os.path.isfile(json_file_path):
            return {'error': f'File not found: {json_file_path}'}
        documents = get_documents(json_file_path)
        if documents is None:
            return {'error': 'Unable to read JSON file'}

        features = extract_features(documents)
        other_features = extract_other(documents)

        output_file = "output.npz"
        prepare(features, other_features, output_file)

    else:
        return {'error': 'Invalid choice'}

    data = load_npz(output_file)

    model = load_model(client, db_name, collection_name, model_name)
    label_encoder = load_encoder(client, db_name2, collection_name2, encoder_name)
    if model:
        predictions = model.predict_proba(data)

        result = []
        for prediction in predictions:
            max_index = np.argmax(prediction)
            original_label = label_encoder.inverse_transform([max_index])[0]
            probability = round(prediction[max_index] * 100, 2)
            result.append({"label": original_label, "probability": f"{probability}%"})

        result.sort(key=lambda x: x['probability'], reverse=True)

        return result[0]
    else:
        return {'error': 'Failed to load model'}

sample_data = {
    "choice": "1",
    "title": "[ST][FDD][SBTS00][Airscale+Airscale][4G-5G][AAFIA][ABIN][CB008993] AAIB RU stuck in initialization state after remove - insert the fiber optic cable from ABIA",
    "description": "*** DEFAULT TEMPLATE for 2G-3G-4G-5G-SRAN-FDD-TDD-DCM-Micro-Controller common template v1.4.0 (02.07.2021) – PLEASE FILL IT IN BEFORE CREATING A PR AND DO NOT CHANGE / REMOVE ANY SECTION OF THIS TEMPLATE ***\r\n\r\n[1. Detail Test Steps:]\r\n \r\n Pre-req:\r\n5G BTS and 4G BTS onAir\r\n4G BTS keeps the primary link \r\n\r\n Steps:\r\n1)Remove the fiber optic cable connected to a shared RU (AAIB) from 4G bbmod ABIA\r\n2)After the RU restarts and 5G cells are onAir, insert back the fiber optic cable into bbmod \r\n\r\n\r\n[2. Expected Result:]\r\n\r\n1) the impacted RU will restart due to primary link lost and alarm with FID:10 should be present on both BTS's\r\n   the RU will be available in 5G BTS and 5G cells will get onAir\r\n2) the RU will get online and 4G cells will be onAir\r\n\r\n[3. Actual Result:]\r\n\r\n1) FID:10 present on both BTSs - OK\r\n   RU is available and 5G cells are onAir - OK\r\n2) the RU is stuck in initialization mode in 4G BTS after FO cable is plugged back in - NOK\r\n   FID:10 is not cleared in 4G BTS - NOK\r\n\r\n\r\n[4. Tester analysis:]\r\n\r\nUnplugged FO cable from ABIA BBMOD_L-3 connected to AAIB RMOD_L-2 (SN:BL2110I27AL)\r\nLTE SYSLOG_102.log\r\n\r\n043628 16.03 16:28:14.151 [192.168.255.129] 04 FCT-1011-3-BTSOMex 2022-03-16T14:28:14.143577Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [CpriOptLink:BBMOD_L-3:CONNECTOR_L-2] handleNonFujitsuLosGuardTimeoutPostDetection loss-of-sync guard timer expired, resynchronizing\r\n043629 16.03 16:28:14.151 [192.168.255.129] 05 FCT-1011-3-BTSOMex 2022-03-16T14:28:14.143584Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [CpriOptLink:BBMOD_L-3:CONNECTOR_L-2] processCpriResynchronizingInPostDetection cpri resynchronizing in post-detection\r\n043630 16.03 16:28:14.151 [192.168.255.129] 06 FCT-1011-3-BTSOMex 2022-03-16T14:28:14.143626Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [CpriOptLink:BBMOD_L-3:CONNECTOR_L-2] 90 s sync timer started\r\n043631 16.03 16:28:14.151 [192.168.255.129] 07 FCT-1011-3-BTSOMex 2022-03-16T14:28:14.143632Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [CpriOptLink:BBMOD_L-3:CONNECTOR_L-2] processLossOfSync lost sync\r\n043632 16.03 16:28:14.151 [192.168.255.129] 08 FCT-1011-3-BTSOMex 2022-03-16T14:28:14.143639Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [PortDetector:BBMOD_L-3:CONNECTOR_L-2] detection lost\r\n043633 16.03 16:28:14.151 [192.168.255.129] 09 FCT-1011-3-BTSOMex 2022-03-16T14:28:14.143662Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [PortDetector:BBMOD_L-3:CONNECTOR_L-2] setting connector state to inactive\r\n\r\nAlarm with FID:10 present\r\n\r\n072079 16.03 16:28:20.581 [192.168.255.129] 86 FCT-1011-3-FRI 2022-03-16T14:28:20.552637Z 459B-FRI INF/FRI/FRI, [AlarmReporter] Activated alarm : /MRBTS-1/RAT-1/FRI-1/ALARM-56, Reported By: /MRBTS-1/RAT-1/RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-5, FaultId: 10, AlarmingDN: /MRBTS-1/RAT-1/RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-5, Severity: 0, AlarmNumber: 7116,\r\n072080 16.03 16:28:20.581 [192.168.255.129] 87 FCT-1011-3-FRI 2022-03-16T14:28:20.552641Z 459B-FRI INF/FRI/FRI, [AlarmReporter] Invisible : 0\r\n072081 16.03 16:28:20.581 [192.168.255.129] 88 FCT-1011-3-FRI 2022-03-16T14:28:20.552650Z 459B-FRI INF/FRI/FRI, [TogglingAlarmDecorator] Set alarm\r\n072082 16.03 16:28:20.581 [192.168.255.129] 89 FCT-1011-3-FRI 2022-03-16T14:28:20.552656Z 459B-FRI INF/FRI/FRI, [TogglingAlarmConfiguration] RMOD_R/10 : Toggling disabled\r\n072083 16.03 16:28:20.581 [192.168.255.129] 8a FCT-1011-3-FRI 2022-03-16T14:28:20.552711Z 459B-FRI INF/FRI/FRI, [TogglingAlarmConfiguration] RMOD_R/10 : Toggling disabled for group3030\r\n072084 16.03 16:28:20.581 [192.168.255.129] 8b FCT-1011-3-FRI 2022-03-16T14:28:20.552752Z 459B-FRI INF/FRI/FRI, [AlarmBuffer] Buffering alarm set\r\n072085 16.03 16:28:20.581 [192.168.255.129] 8c FCT-1011-3-FRI 2022-03-16T14:28:20.552837Z 459B-FRI INF/FRI/FRI, [AlarmBuffer] Clearing buffer\r\n072086 16.03 16:28:20.581 [192.168.255.129] 8d FCT-1011-3-FRI 2022-03-16T14:28:20.553470Z 459B-FRI INF/FRI/FRI, [SharedInfoUpdater] Set alarm: /MRBTS-1/RAT-1/FRI-1/ALARM-56\r\n072087 16.03 16:28:20.581 [192.168.255.129] 8e FCT-1011-3-FRI 2022-03-16T14:28:20.553518Z 459B-FRI INF/FRI/FRI, [SnapshotConfiguration] Fault is in group: 3030\r\n\r\nRMOD_L-2 available and cells onAir on 5G:\r\n5G syslog_72.log\r\n\r\n107923 16.03 16:33:02.439 [192.168.255.129] 5d FCT-2011-3-MCtrl 2022-03-16T14:33:01.639056Z 2B1F-MCtrl INF/MCtrl/CM, [CellDriver] StateInfo::availabilityStatus of /MRBTS-1/RAT-1/MCTRL-4/BBTOP_M-1/MRBTS_M-1/NRBTS_M-1/CELL_M-12 has been changed from Offline to Online\r\n115066 16.03 16:33:05.450 [192.168.255.129] c9 FCT-2011-3-MCtrl 2022-03-16T14:33:03.741246Z 2B1F-MCtrl INF/MCtrl/CM, [CellDriver] StateInfo::availabilityStatus of /MRBTS-1/RAT-1/MCTRL-4/BBTOP_M-1/MRBTS_M-1/NRBTS_M-1/CELL_M-13 has been changed from Offline to Online\r\n\r\nPlugged in the FO cable:\r\n4G syslog_103.log\r\n\r\n351445 16.03 16:33:39.561 [192.168.255.129] 13 FCT-1011-3-BTSOMex 2022-03-16T14:33:39.560587Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [VsbMonitor:BBMOD_L-3:CONNECTOR_L-2] received API_CPRI_VSB_DATA_IND_MSG [cpriLink 1, refNo 0, data [{ #16 00 } ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff { #16 00 } ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff { #16 00 } ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff ff { #16 00 } .^.\r\n351446 16.03 16:33:39.561 [192.168.255.129] 14 FCT-1011-3-BTSOMex 2022-03-16T14:33:39.560705Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [CpriOptLink:BBMOD_L-3:CONNECTOR_L-2] received VsbRxData indication\r\n\r\n\r\nRadio detected but stuck in initialization state:\r\n\r\n357053 16.03 16:33:41.154 [192.168.255.129] 3c FCT-1011-3-BTSOMex 2022-03-16T14:33:40.280697Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [VsbMonitor:BBMOD_L-3:CONNECTOR_L-2] sendSendVsbDataRequest sending API_CPRI_SEND_VSB_DATA_REQ_MSG [cpriLink 1, refNo 0, repeatOn 1, pattern 0, data [fe fe { #46 00 }]]\r\n357054 16.03 16:33:41.154 [192.168.255.129] 3d FCT-1011-3-BTSOMex 2022-03-16T14:33:40.280788Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [CpriOptLink:BBMOD_L-3:CONNECTOR_L-2] received VsbRxData indication\r\n357055 16.03 16:33:41.154 [192.168.255.129] 3e FCT-1011-3-BTSOMex 2022-03-16T14:33:40.280794Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [CpriOptLink:BBMOD_L-3:CONNECTOR_L-2] Nokia radio already detected. Master discovery acknowledgement received.\r\n357056 16.03 16:33:41.154 [192.168.255.129] 3f FCT-1011-3-BTSOMex 2022-03-16T14:33:40.280936Z 4742-MOAM_MAIN INF/LGC/FEAT_DBG_KNIFE_COMPATIBLE_Disable, [CpriOptLink:BBMOD_L-3:CONNECTOR_L-2] PsuAlarm is not supported\r\n357057 16.03 16:33:41.154 [192.168.255.129] 40 FCT-1011-3-HAS 2022-03-16T14:33:40.282585Z 47E1-BB_SYNC INF/Has/BB_SYNCHRONIZATION, [PmCountersService::NtpCounter] reportValues: NTP_L does not exist or systemStatistics not set\r\n357058 16.03 16:33:41.154 [192.168.255.129] 41 FCT-1011-3-DEM 2022-03-16T14:33:40.282718Z 45A9-DEM INF/Dem/Rem/Service, [rvs::RmodStateUpdater] setAdministrativeState: set lock status for slave, rmodr: RUNTIME_VIEW-1/MRBTS_R-1/EQM_R-1/APEQM_R-1/RMOD_R-5\r\n357059 16.03 16:33:41.154 [192.168.255.129] 42 FCT-1011-3-DEM 2022-03-16T14:33:40.283017Z 45A9-DEM INF/Dem/Rem/Service, [rvs::RmodStateUpdater] handleChange: triggered by: BTS_L-1/EQM_L-1/RMOD_L-2\r\n\r\nI have assigned to BOAM_CA_FRONTHAUL due to PCI results and other similar but not the same PRs that are handled by BOAM_FH_DCMM\r\n\r\n[5. Pronto Creator Interface (PCI) information:]\r\n\r\nPCI executed for Problem Report (17.03.2022, 20:34:32 UTC):\r\nPCI ID: PCI_20220317_212738_000\r\nMLoGIC model version: 1.0.0-alpha\r\nAccess PCI to check saved results: https://rep-portal.wroclaw.nsn-rdnet.net/pci/?pci_id=PCI_20220317_212738_000\r\n\r\n[6. Log(s) file name containing a fault: (clear indication (exact file name) and timestamp where fault can be found in attached logs):]\r\n\r\n\\\\eseefsn50.emea.nsn-net.net\\rotta4internal\\LTE_2\\solah\\RU_stuck_init\r\n\r\n\r\n[7. Test-Line Reference/used HW/configuration/tools/SW version:]\r\n\r\n5G SW: SBTS22R3_ENB_9999_220315_000002\r\n5G HW: ASIL + ABIN, 3xAAIB shared, 6x5MHz 4T4R B66, mMIMO\r\nSBTS SW: SBTS22R3_ENB_9999_220315_000002\r\nSBTS HW: 2xASIB + 6xABIA, 3xAAIB shared, 3xAAFB dedicated 1xAHLOA, 6x15MHz 4T4R B66 mMIMO, 6x20MHz 4T4R B25 mMIMO, 1x10MHz anchor B12 4T4R\r\n\r\n[8. Used Flags: (list here used R&D flags):]\r\n\r\nNR\r\n\r\n### Enable Firewall feature 5GC000265 & [5GC000325, 5GC000268]  ###\r\n0x1A1300=0 #1=enabled (default) ; 0=disabled\r\n### V2.2 to set K2MIN = 2 \"\"\"ONLY VIAVI TESTLINES\"\"\" Ticket ubi00142986. Restriction reported by Viavi to apply K2min=1 ##\r\n0x58012D=2\r\n# BTSLOG\r\n0x1003F=1          # AaSysLogInputLevel  0x2\r\n0x10040=1          # AaSysLogOutputLevel 0x2\r\n#0x10041=5          # AaSysLogOutputMode  0 = none ; 1 = all ; 2 = udp ; 3 = sic ; 4 = standard output ; 5 = local ; 6 = remote ; 7 = not valid\r\n0x10042=0xC0A8FF7E # AaSysLogUdpAddress  192.168.255.126\r\n0x10043=0xC738     # AaSysLogUdpPort     51000\r\n#0x4F00E1=1  #SPMAG\r\n#0x420026=1  #DCS\r\n#OAM_FEAT_RUMA_Enable_5GC00933, RAD_OAM(0x021C), 1)\r\n#0x021C=1\r\n0x45005a=1          # enable developer user login\r\n0x420000=1\r\n#0x190030=0\r\n#BBC debug and verbose flags:\r\n275=1\r\n0x19011E=1\r\n# Le\r\n0x10041=1   \r\n0x1002A = 2",
    "build": "SBTS00_ENB_9999_220315_000002",
    "feature": "CB008993-SR-A-B5",
    "release": "SBTS00"
}

prediction = predict(sample_data)
print(prediction)
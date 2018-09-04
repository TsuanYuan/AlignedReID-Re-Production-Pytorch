# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tracking_results.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


import common_pb2 as common__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tracking_results.proto',
  package='tracking_pipeline',
  syntax='proto3',
  serialized_pb=_b('\n\x16tracking_results.proto\x12\x11tracking_pipeline\x1a\x0c\x63ommon.proto\"\xae\x01\n\x0fTrackingResults\x12\x16\n\x0e\x64\x65tection_file\x18\x01 \x01(\t\x12\x12\n\nvideo_path\x18\x02 \x01(\t\x12\x1c\n\x14\x64\x65tection_confidence\x18\x03 \x01(\x02\x12@\n\x12tracked_detections\x18\x04 \x03(\x0b\x32$.tracking_pipeline.TrackingDetection\x12\x0f\n\x07version\x18\x05 \x01(\t\"U\n\x16StaffRecognitionResult\x12\x16\n\x0estaff_recog_id\x18\x01 \x01(\x05\x12\x0f\n\x07\x66\x61\x63\x65_id\x18\x02 \x01(\x05\x12\x12\n\nconfidence\x18\x03 \x01(\x02\"J\n\x19\x43ustomerRecognitionResult\x12\x19\n\x11\x63ustomer_recog_id\x18\x01 \x01(\x05\x12\x12\n\nconfidence\x18\x02 \x01(\x02\"Q\n\x0eLabelingResult\x12\x18\n\x10pure_tracklet_id\x18\x01 \x01(\x05\x12\x18\n\x10long_tracklet_id\x18\x02 \x01(\x05\x12\x0b\n\x03pid\x18\x03 \x01(\x05\"\xe0\x05\n\x11TrackingDetection\x12\x12\n\nvideo_name\x18\x14 \x01(\t\x12\x13\n\x0b\x66rame_index\x18\x01 \x01(\r\x12\x16\n\x0etracklet_index\x18\x02 \x01(\r\x12\x17\n\x0ftrack_confirmed\x18\x03 \x01(\x08\x12\x14\n\x0c\x62ody_visible\x18\x04 \x01(\x08\x12/\n\nkey_points\x18\x05 \x03(\x0b\x32\x1b.tracking_pipeline.KeyPoint\x12\x18\n\x10\x62ody_kf_velocity\x18\x06 \x03(\x02\x12\x1d\n\x15human_detection_score\x18\x07 \x01(\x02\x12\x13\n\x0bhuman_box_x\x18\x08 \x01(\x05\x12\x13\n\x0bhuman_box_y\x18\t \x01(\x05\x12\x18\n\x10human_box_height\x18\n \x01(\x05\x12\x17\n\x0fhuman_box_width\x18\x0b \x01(\x05\x12\x12\n\nhead_box_x\x18\x0c \x01(\x05\x12\x12\n\nhead_box_y\x18\r \x01(\x05\x12\x17\n\x0fhead_box_height\x18\x0e \x01(\x05\x12\x16\n\x0ehead_box_width\x18\x0f \x01(\x05\x12\x1a\n\x12\x61ppearance_feature\x18\x10 \x01(\x0c\x12\x16\n\x0esv_tracklet_id\x18\x17 \x01(\x05\x12\x14\n\x0cmv_linked_id\x18\x11 \x01(\x05\x12\x15\n\rface_recog_id\x18\x12 \x01(\x05\x12K\n\x18staff_recognition_result\x18\x13 \x01(\x0b\x32).tracking_pipeline.StaffRecognitionResult\x12:\n\x0flabeling_result\x18\x15 \x01(\x0b\x32!.tracking_pipeline.LabelingResult\x12Q\n\x1b\x63ustomer_recognition_result\x18\x16 \x01(\x0b\x32,.tracking_pipeline.CustomerRecognitionResultb\x06proto3')
  ,
  dependencies=[common__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_TRACKINGRESULTS = _descriptor.Descriptor(
  name='TrackingResults',
  full_name='tracking_pipeline.TrackingResults',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='detection_file', full_name='tracking_pipeline.TrackingResults.detection_file', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='video_path', full_name='tracking_pipeline.TrackingResults.video_path', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='detection_confidence', full_name='tracking_pipeline.TrackingResults.detection_confidence', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tracked_detections', full_name='tracking_pipeline.TrackingResults.tracked_detections', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='version', full_name='tracking_pipeline.TrackingResults.version', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=234,
)


_STAFFRECOGNITIONRESULT = _descriptor.Descriptor(
  name='StaffRecognitionResult',
  full_name='tracking_pipeline.StaffRecognitionResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='staff_recog_id', full_name='tracking_pipeline.StaffRecognitionResult.staff_recog_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='face_id', full_name='tracking_pipeline.StaffRecognitionResult.face_id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='tracking_pipeline.StaffRecognitionResult.confidence', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=236,
  serialized_end=321,
)


_CUSTOMERRECOGNITIONRESULT = _descriptor.Descriptor(
  name='CustomerRecognitionResult',
  full_name='tracking_pipeline.CustomerRecognitionResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='customer_recog_id', full_name='tracking_pipeline.CustomerRecognitionResult.customer_recog_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='confidence', full_name='tracking_pipeline.CustomerRecognitionResult.confidence', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=323,
  serialized_end=397,
)


_LABELINGRESULT = _descriptor.Descriptor(
  name='LabelingResult',
  full_name='tracking_pipeline.LabelingResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='pure_tracklet_id', full_name='tracking_pipeline.LabelingResult.pure_tracklet_id', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='long_tracklet_id', full_name='tracking_pipeline.LabelingResult.long_tracklet_id', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pid', full_name='tracking_pipeline.LabelingResult.pid', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=399,
  serialized_end=480,
)


_TRACKINGDETECTION = _descriptor.Descriptor(
  name='TrackingDetection',
  full_name='tracking_pipeline.TrackingDetection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='video_name', full_name='tracking_pipeline.TrackingDetection.video_name', index=0,
      number=20, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='frame_index', full_name='tracking_pipeline.TrackingDetection.frame_index', index=1,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='tracklet_index', full_name='tracking_pipeline.TrackingDetection.tracklet_index', index=2,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='track_confirmed', full_name='tracking_pipeline.TrackingDetection.track_confirmed', index=3,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='body_visible', full_name='tracking_pipeline.TrackingDetection.body_visible', index=4,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='key_points', full_name='tracking_pipeline.TrackingDetection.key_points', index=5,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='body_kf_velocity', full_name='tracking_pipeline.TrackingDetection.body_kf_velocity', index=6,
      number=6, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='human_detection_score', full_name='tracking_pipeline.TrackingDetection.human_detection_score', index=7,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='human_box_x', full_name='tracking_pipeline.TrackingDetection.human_box_x', index=8,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='human_box_y', full_name='tracking_pipeline.TrackingDetection.human_box_y', index=9,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='human_box_height', full_name='tracking_pipeline.TrackingDetection.human_box_height', index=10,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='human_box_width', full_name='tracking_pipeline.TrackingDetection.human_box_width', index=11,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='head_box_x', full_name='tracking_pipeline.TrackingDetection.head_box_x', index=12,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='head_box_y', full_name='tracking_pipeline.TrackingDetection.head_box_y', index=13,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='head_box_height', full_name='tracking_pipeline.TrackingDetection.head_box_height', index=14,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='head_box_width', full_name='tracking_pipeline.TrackingDetection.head_box_width', index=15,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='appearance_feature', full_name='tracking_pipeline.TrackingDetection.appearance_feature', index=16,
      number=16, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='sv_tracklet_id', full_name='tracking_pipeline.TrackingDetection.sv_tracklet_id', index=17,
      number=23, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mv_linked_id', full_name='tracking_pipeline.TrackingDetection.mv_linked_id', index=18,
      number=17, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='face_recog_id', full_name='tracking_pipeline.TrackingDetection.face_recog_id', index=19,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='staff_recognition_result', full_name='tracking_pipeline.TrackingDetection.staff_recognition_result', index=20,
      number=19, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='labeling_result', full_name='tracking_pipeline.TrackingDetection.labeling_result', index=21,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='customer_recognition_result', full_name='tracking_pipeline.TrackingDetection.customer_recognition_result', index=22,
      number=22, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=483,
  serialized_end=1219,
)

_TRACKINGRESULTS.fields_by_name['tracked_detections'].message_type = _TRACKINGDETECTION
_TRACKINGDETECTION.fields_by_name['key_points'].message_type = common__pb2._KEYPOINT
_TRACKINGDETECTION.fields_by_name['staff_recognition_result'].message_type = _STAFFRECOGNITIONRESULT
_TRACKINGDETECTION.fields_by_name['labeling_result'].message_type = _LABELINGRESULT
_TRACKINGDETECTION.fields_by_name['customer_recognition_result'].message_type = _CUSTOMERRECOGNITIONRESULT
DESCRIPTOR.message_types_by_name['TrackingResults'] = _TRACKINGRESULTS
DESCRIPTOR.message_types_by_name['StaffRecognitionResult'] = _STAFFRECOGNITIONRESULT
DESCRIPTOR.message_types_by_name['CustomerRecognitionResult'] = _CUSTOMERRECOGNITIONRESULT
DESCRIPTOR.message_types_by_name['LabelingResult'] = _LABELINGRESULT
DESCRIPTOR.message_types_by_name['TrackingDetection'] = _TRACKINGDETECTION

TrackingResults = _reflection.GeneratedProtocolMessageType('TrackingResults', (_message.Message,), dict(
  DESCRIPTOR = _TRACKINGRESULTS,
  __module__ = 'tracking_results_pb2'
  # @@protoc_insertion_point(class_scope:tracking_pipeline.TrackingResults)
  ))
_sym_db.RegisterMessage(TrackingResults)

StaffRecognitionResult = _reflection.GeneratedProtocolMessageType('StaffRecognitionResult', (_message.Message,), dict(
  DESCRIPTOR = _STAFFRECOGNITIONRESULT,
  __module__ = 'tracking_results_pb2'
  # @@protoc_insertion_point(class_scope:tracking_pipeline.StaffRecognitionResult)
  ))
_sym_db.RegisterMessage(StaffRecognitionResult)

CustomerRecognitionResult = _reflection.GeneratedProtocolMessageType('CustomerRecognitionResult', (_message.Message,), dict(
  DESCRIPTOR = _CUSTOMERRECOGNITIONRESULT,
  __module__ = 'tracking_results_pb2'
  # @@protoc_insertion_point(class_scope:tracking_pipeline.CustomerRecognitionResult)
  ))
_sym_db.RegisterMessage(CustomerRecognitionResult)

LabelingResult = _reflection.GeneratedProtocolMessageType('LabelingResult', (_message.Message,), dict(
  DESCRIPTOR = _LABELINGRESULT,
  __module__ = 'tracking_results_pb2'
  # @@protoc_insertion_point(class_scope:tracking_pipeline.LabelingResult)
  ))
_sym_db.RegisterMessage(LabelingResult)

TrackingDetection = _reflection.GeneratedProtocolMessageType('TrackingDetection', (_message.Message,), dict(
  DESCRIPTOR = _TRACKINGDETECTION,
  __module__ = 'tracking_results_pb2'
  # @@protoc_insertion_point(class_scope:tracking_pipeline.TrackingDetection)
  ))
_sym_db.RegisterMessage(TrackingDetection)


# @@protoc_insertion_point(module_scope)

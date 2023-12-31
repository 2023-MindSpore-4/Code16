#!/usr/bin/env python
# coding=utf-8
"""
Function:
SubCaseDesignCross class
This class mainly involves parse
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import itertools

from op_test_frame.st.interface.const_manager import ConstManager
from op_test_frame.st.interface import utils
from op_test_frame.st.interface import dynamic_handle
from op_test_frame.st.interface import subcase_design as SD


def combine_ori_field_to_cross(tensor, cross_key_list):
    """
    combine ori_field to cross for subcase
    :param tensor: the key of json object
    :param cross_key_list: the list of cross keys
    :return: none
    """
    cross_list = []
    ori_field_cross_key_list = []
    for key in cross_key_list:
        # copy cross_key_list to ori_field_cross_key_list
        ori_field_cross_key_list.append(key)
        if key in ['format', 'shape']:
            continue
        cross_list.append(tensor.get(key))
    # insert ori_format, and ori_format needs to close format
    ori_field_cross_key_list.insert(1, 'ori_format')
    # orthogonal combination of format and ori_format
    combine_format_ori_format_list = list(
        zip(tensor.get('format'), tensor.get('ori_format')))
    # insert ori_shape, and ori_shape needs to close shape
    ori_field_cross_key_list.insert(3, 'ori_shape')
    # orthogonal combination of shape and ori_shape
    combine_shape_ori_shape_list = list(
        zip(tensor.get('shape'), tensor.get('ori_shape')))
    # orthonormalize format_ori_format, shape_ori_shape, and other filed: 'type', etc.
    orthonormalize_itertools = itertools.product(combine_format_ori_format_list,
                                                 combine_shape_ori_shape_list, *cross_list)
    combine_cross_tuple = (list(x) for x in orthonormalize_itertools)
    result_cross_list = []
    for each_cross_list in list(combine_cross_tuple):
        data_list = _get_data_list(each_cross_list)
        result_cross_list.append(data_list)
    return ori_field_cross_key_list, result_cross_list


def _get_data_list(each_cross_list):
    data_list = []
    for filed_data in each_cross_list:
        if isinstance(filed_data, tuple):
            for data in filed_data:
                data_list.append(data)
        else:
            data_list.append(filed_data)
    return data_list


class SubCaseDesignCross(SD.SubCaseDesign):
    """
    the class for design test subcase by cross.
    """

    def __init__(self, current_json_path, json_obj,
                 total_case_list, report):
        super(SubCaseDesignCross, self).__init__(current_json_path, json_obj,
                                                 total_case_list, report)
        self.multi = False

    @staticmethod
    def _check_input_count(case_list, key_desc):
        if len(set(len(i) for i in case_list)) != 1:
            utils.print_error_log("The length of %s is inconsistent in operator information description" % key_desc)
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    @staticmethod
    def _append_input_desc_to_case(json_obj, index, input_case_list, case):
        # check input_case_list for support no input scene
        if len(input_case_list) < 1:
            return
        for input_index, input_case in enumerate(input_case_list):
            if json_obj[ConstManager.INPUT_DESC][input_index].get('name'):
                input_name = \
                    json_obj[ConstManager.INPUT_DESC][input_index].get('name')
                input_case[index].update({'name': input_name})

            case[ConstManager.INPUT_DESC].append(input_case[index])

    def check_number_match(self, key, count, desc_list):
        """
        check number match
        :param key: the key of json object
        :param count: correct count
        :param desc_list: desc list
        :return: none
        """
        for item in desc_list:
            if count != len(item[key]):
                utils.print_error_log(
                    'The number of "%s" of the inputs must be consistent with '
                    'the number of "%s" of the outputs in %s. Please modify.'
                    % (key, key, self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    def subcase_generate(self):
        """
        generate subcase by cross
        :return: the test case list
        """
        if self.json_obj.get(ConstManager.ST_MODE) == "ms_python_train":
            input_desc_list = self._make_input_desc_list_ms(self.json_obj)
            output_desc_list = self._make_output_desc_list_ms(self.json_obj)
            attr_list = self._check_attr_valid(self.json_obj)
            self._check_value_number_valid_ms(input_desc_list,
                                              output_desc_list)
            input_case_list = self._cross_tensor(
                input_desc_list, ConstManager.MS_INPUT_CROSS_LIST)
            output_case_list = self._cross_tensor(
                output_desc_list, ConstManager.MS_OUTPUT_CROSS_LIST)
        else:
            input_desc_list = self._make_input_desc_list(self.json_obj)
            output_desc_list = self._make_output_desc_list(self.json_obj)
            attr_list = self._check_attr_valid(self.json_obj)
            self._check_value_number_valid(
                input_desc_list, output_desc_list)
            input_case_list = self._cross_tensor(
                input_desc_list, ConstManager.INPUT_CROSS_LIST)
            output_case_list = self._cross_tensor(
                output_desc_list, ConstManager.OUTPUT_CROSS_LIST)
        count = self._get_count(input_case_list, output_case_list)
        prefix = '{}{}'.format(self.json_obj.get(ConstManager.CASE_NAME).replace('/', '_'), '_')
        if self.multi:
            if self.json_obj.get(ConstManager.ST_MODE) == "ms_python_train":
                prefix += 'sub_'
            else:
                prefix += 'sub_case_'
        else:
            prefix += 'case_'
        case_dict = {'input': input_case_list,
                     'output': output_case_list}
        self._get_sub_test_cases(case_dict, count, prefix, attr_list)
        return self.total_case_list

    def _get_count(self, input_case_list, output_case_list):
        # for support no inputs
        if len(input_case_list) > 1:
            self._check_input_count(input_case_list, 'input')
            return len(input_case_list[0])
        self._check_input_count(output_case_list, 'output')
        return len(output_case_list[0])

    def _get_data_distribute_list(self, input_desc):
        if 'data_distribute' in input_desc and ConstManager.VALUE not in input_desc:
            data_distribute_list = self._check_list_str_valid(
                input_desc, 'data_distribute',
                self.WHITE_LISTS.data_distribution_list, ConstManager.INPUT_DESC)
        else:
            data_distribute_list = ['uniform']
        return data_distribute_list

    def _get_value_range_list(self, input_desc):
        if 'value_range' in input_desc and ConstManager.VALUE not in input_desc:
            value_range_list = self._check_list_list_valid(
                input_desc, 'value_range', ConstManager.INPUT_DESC)
            for item in value_range_list:
                self._check_range_value_valid(item)
        else:
            value_range_list = [[0.1, 1.0]]
        return value_range_list

    def _make_input_desc_list_ms(self, json_obj):
        input_desc_list = []
        if len(json_obj[ConstManager.INPUT_DESC]) == 0:
            utils.print_error_log(
                'The value of "input_desc" is empty. Please modify it in '
                'file %s.' % self.current_json_path)
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for input_desc in json_obj[ConstManager.INPUT_DESC]:
            type_list = self._check_list_str_valid(
                input_desc, 'type', self.WHITE_LISTS.mindspore_type_list,
                ConstManager.INPUT_DESC)
            shape_list = self._check_list_list_valid(
                input_desc, 'shape', ConstManager.INPUT_DESC)
            for item in shape_list:
                self._check_shape_valid(item)
            data_distribute_list = self._get_data_distribute_list(input_desc)
            value_range_list = self._get_value_range_list(input_desc)
            one_input_desc = {'type': type_list,
                              'shape': shape_list,
                              'value_range': value_range_list,
                              'data_distribute': data_distribute_list}
            self._deal_with_value(input_desc, one_input_desc)
            input_desc_list.append(one_input_desc)
            for item in one_input_desc.values():
                if len(item) > 1:
                    self.multi = True
        return input_desc_list

    def _make_output_desc_list_ms(self, json_obj):
        output_desc_list = []
        if len(json_obj[ConstManager.OUTPUT_DESC]) == 0:
            utils.print_error_log(
                'The value of "output_desc" is empty. Please modify it in '
                'file %s.' % self.current_json_path)
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for output_desc in json_obj[ConstManager.OUTPUT_DESC]:
            type_list = self._check_list_str_valid(
                output_desc, 'type', self.WHITE_LISTS.mindspore_type_list,
                ConstManager.OUTPUT_DESC)
            shape_list = self._check_list_list_valid(
                output_desc, 'shape', ConstManager.OUTPUT_DESC)
            for item in shape_list:
                self._check_shape_valid(item)
            one_output_desc = {'type': type_list,
                               'shape': shape_list}
            output_desc_list.append(one_output_desc)
            for item in one_output_desc.values():
                if len(item) > 1:
                    self.multi = True
        return output_desc_list

    def _check_value_number_valid_ms(self, input_desc_list, output_desc_list):
        key = 'type'
        count = len(input_desc_list[0][key])
        self.check_number_match(key, count, input_desc_list)
        self.check_number_match(key, count, output_desc_list)
        key = 'shape'
        count = len(input_desc_list[0][key])
        self.check_number_match(key, count, input_desc_list)
        self.check_number_match(key, count, output_desc_list)

    def _check_value_number_valid(self, input_desc_list, output_desc_list):
        if len(input_desc_list) < 1:
            return
        key = 'format'
        count = len(input_desc_list[0][key])
        self.check_number_match(key, count, input_desc_list)
        self.check_number_match(key, count, output_desc_list)
        key = 'type'
        count = len(input_desc_list[0][key])
        self.check_number_match(key, count, input_desc_list)
        self.check_number_match(key, count, output_desc_list)
        key = 'shape'
        count = len(input_desc_list[0][key])
        self.check_number_match(key, count, input_desc_list)
        self.check_number_match(key, count, output_desc_list)

    def _get_case_list(self, tensor, cross_list, cross_key_list):
        case_list = []
        for case in cross_list:
            cur_params = {cross_key_list[x]: case[x] for x, _ in enumerate(cross_key_list)}
            self._check_cur_params_undefined(cur_params)
            if cur_params.get('shape'):
                dynamic_handle.set_typical_shape_in_cur_params(
                    cur_params, tensor, self.current_json_path)
            case_list.append(cur_params)
        return case_list

    def _cross_tensor(self, tensor_list, op_cross_key_list):
        total_case_list = []
        for tensor in tensor_list:
            cross_list = []
            cross_key_list = []
            for key in op_cross_key_list:
                cross_key_list.append(key)
            # add new key in cross_key_list
            utils.add_new_key_to_cross_list(tensor, cross_key_list)
            if tensor.get('ori_format') and tensor.get('ori_shape'):
                ori_field_cross_key_list, result_cross_list = \
                    combine_ori_field_to_cross(tensor, cross_key_list)
                case_list = self._get_case_list(tensor, result_cross_list,
                                                ori_field_cross_key_list)
            else:
                for key in cross_key_list:
                    cross_list.append(tensor[key])
                cross_tuple = (list(x) for x in itertools.product(*cross_list))
                cross_list = list(cross_tuple)
                case_list = self._get_case_list(tensor, cross_list, cross_key_list)
            total_case_list.append(case_list)
        return total_case_list

    def _make_input_desc_list(self, json_obj):
        input_desc_list = []
        if len(json_obj[ConstManager.INPUT_DESC]) == 0:
            utils.print_warn_log(
                'The value of "input_desc" is empty.')
            return input_desc_list
        for input_desc in json_obj[ConstManager.INPUT_DESC]:
            format_list = self._check_list_str_valid(
                input_desc, 'format', list(self.WHITE_LISTS.format_map.keys()), ConstManager.INPUT_DESC)
            type_list = self._check_list_str_valid(
                input_desc, 'type', self.WHITE_LISTS.type_list,
                ConstManager.INPUT_DESC)
            shape_list = self._check_list_list_valid(
                input_desc, 'shape', ConstManager.INPUT_DESC)
            for item in shape_list:
                self._check_shape_valid(item)
            data_distribute_list = self._get_data_distribute_list(input_desc)
            value_range_list = self._get_value_range_list(input_desc)
            ori_filed_list = self._get_ori_filed_data(input_desc, ConstManager.INPUT_DESC, format_list, shape_list)
            if ori_filed_list:
                # add ori_format and ori_shape for one_input_desc
                one_input_desc = {'format': format_list,
                                  'ori_format': ori_filed_list[0],
                                  'type': type_list,
                                  'shape': shape_list,
                                  'ori_shape': ori_filed_list[1],
                                  'value_range': value_range_list,
                                  'data_distribute': data_distribute_list}
            else:
                one_input_desc = {'format': format_list, 'type': type_list,
                                  'shape': shape_list,
                                  'value_range': value_range_list,
                                  'data_distribute': data_distribute_list}
            # check whether the shape is dynamic.
            if input_desc.get(
                    ConstManager.TYPICAL_SHAPE) is not None:
                self._get_dynamic_shape_info(
                    ConstManager.INPUT_DESC, input_desc, one_input_desc)

            self._deal_with_value(input_desc, one_input_desc)

            input_desc_list.append(one_input_desc)
            for item in one_input_desc.values():
                if len(item) > 1:
                    self.multi = True
        return input_desc_list

    def _make_output_desc_list(self, json_obj):
        output_desc_list = []
        if len(json_obj[ConstManager.OUTPUT_DESC]) == 0:
            utils.print_error_log(
                'The value of "output_desc" is empty. Please modify it in '
                'file %s.' % self.current_json_path)
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for output_desc in json_obj[ConstManager.OUTPUT_DESC]:
            format_list = self._check_list_str_valid(
                output_desc, 'format', list(self.WHITE_LISTS.format_map.keys()), ConstManager.OUTPUT_DESC)
            type_list = self._check_list_str_valid(
                output_desc, 'type', self.WHITE_LISTS.type_list,
                ConstManager.OUTPUT_DESC)
            shape_list = self._check_list_list_valid(
                output_desc, 'shape', ConstManager.OUTPUT_DESC)
            for item in shape_list:
                self._check_shape_valid(item)
            ori_filed_list = self._get_ori_filed_data(output_desc, ConstManager.OUTPUT_DESC, format_list, shape_list)
            if ori_filed_list:
                # add ori_format and ori_shape for one_output_desc
                one_output_desc = {'format': format_list,
                                   'ori_format': ori_filed_list[0],
                                   'type': type_list,
                                   'shape': shape_list,
                                   'ori_shape': ori_filed_list[1]}
            else:
                one_output_desc = {'format': format_list, 'type': type_list,
                                   'shape': shape_list}
            # check whether the shape is dynamic.
            if output_desc.get(
                    ConstManager.TYPICAL_SHAPE) is not None:
                self._get_dynamic_shape_info(
                    ConstManager.OUTPUT_DESC, output_desc, one_output_desc)

            output_desc_list.append(one_output_desc)
            for item in one_output_desc.values():
                if len(item) > 1:
                    self.multi = True
        return output_desc_list

    def _get_sub_test_cases(
            self, case_dict, count, prefix, attr_list):
        pyfile, function = self._check_expect_output_param(self.json_obj)
        err_thr = self._check_set_error_threshold(self.json_obj)
        for index in range(count):
            if self.json_obj.get(ConstManager.ST_MODE) == "ms_python_train":
                case = {ConstManager.OP: self.json_obj[ConstManager.OP],
                        ConstManager.ST_MODE: self.json_obj[ConstManager.ST_MODE],
                        ConstManager.INPUT_DESC: [], ConstManager.OUTPUT_DESC: []}
            else:
                case = {ConstManager.OP: self.json_obj[ConstManager.OP],
                        ConstManager.INPUT_DESC: [], ConstManager.OUTPUT_DESC: []}
            if len(attr_list) > 0:
                case[ConstManager.ATTR] = attr_list
            self._append_input_desc_to_case(self.json_obj, index,
                                            case_dict.get('input'), case)
            output_index = index
            if index >= len(case_dict.get('output')[0]):
                output_index = index % len(case_dict.get('output'))
            for out_index, output_case in enumerate(case_dict.get('output')):
                if self.json_obj[ConstManager.OUTPUT_DESC][out_index].get('name'):
                    output_name = \
                        self.json_obj[ConstManager.OUTPUT_DESC][out_index].get('name')
                    output_case[index].update({'name': output_name})
                case[ConstManager.OUTPUT_DESC].append(output_case[output_index])
            type_str = case[ConstManager.OUTPUT_DESC][0].get('type')
            if self.json_obj.get(ConstManager.ST_MODE) == "ms_python_train":
                suffix_list = ['', type_str]
                suffix = '_'.join(suffix_list)
                case['case_name'] = prefix + '%d' % self.case_idx + suffix
            else:
                format_str = case[ConstManager.OUTPUT_DESC][0].get('format')
                suffix_list = ['', format_str, type_str]
                suffix = '_'.join(suffix_list)
                case['case_name'] = prefix + '%03d' % self.case_idx + suffix
            self.case_idx, self.total_case_list = self._add_case_to_total_case(
                case, self.case_idx, [pyfile, function, err_thr], self.total_case_list)
        utils.print_info_log('Create %d sub test cases for %s.'
                             % (count, self.json_obj[ConstManager.CASE_NAME]))

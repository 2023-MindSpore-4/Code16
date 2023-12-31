#!/usr/bin/env python
# coding=utf-8
"""
Function:
SubCaseDesignFuzz class
This class mainly involves parse
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2020
"""

import os
import sys
import importlib
import copy

from op_test_frame.st.interface import utils
from op_test_frame.st.interface import dynamic_handle
from op_test_frame.st.interface import subcase_design as SD
from op_test_frame.st.interface.const_manager import ConstManager


class SubCaseDesignFuzz(SD.SubCaseDesign):
    """
    the class for design test subcase by fuzz.
    """

    @staticmethod
    def get_fuzz_func_return(fuzz_func):
        """
        Get fuzz func
        """
        fuzz_return_list = []
        fuzz_return = fuzz_func()
        if isinstance(fuzz_return, dict):
            fuzz_return_list.append(fuzz_return)
        elif isinstance(fuzz_return, list):
            fuzz_return_list = fuzz_return
        else:
            utils.print_error_log(
                'The fuzz function return value must be dict or list')
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        return fuzz_return_list

    @staticmethod
    def _get_fuzz_func(module_name, fuzz_func, real_fuzz_path):
        module = importlib.import_module(module_name)
        if not hasattr(module, fuzz_func):
            raise utils.OpTestGenException(
                '%s has no attribute "%s"' % (real_fuzz_path, fuzz_func))
        return getattr(module, fuzz_func)

    def subcase_generate(self):
        """
        generate subcase by cross
        :return: the test case list
        """
        fuzz_function = self.check_fuzz_valid(self.json_obj)
        loop_num = self._check_fuzz_case_num_valid(self.json_obj)
        prefix = '{}{}'.format(
            self.json_obj.get(ConstManager.CASE_NAME).replace('/', '_'),
            '_fuzz_case_')
        pyfile, function = self._check_expect_output_param(self.json_obj)
        err_thr = self._check_set_error_threshold(self.json_obj)
        repeat_case_num = 0
        for _ in range(loop_num):
            ori_json = copy.deepcopy(self.json_obj)
            fuzz_return_list = self.get_fuzz_func_return(fuzz_function)
            for fuzz_dict in fuzz_return_list:
                if ori_json.get(ConstManager.ST_MODE) == "ms_python_train":
                    input_desc_list = self._make_desc_list_ms_fuzz(ori_json,
                                                                   fuzz_dict,
                                                                   ConstManager.INPUT_DESC)
                    output_desc_list = self._make_desc_list_ms_fuzz(ori_json,
                                                                    fuzz_dict,
                                                                    ConstManager.OUTPUT_DESC)
                else:
                    input_desc_list = self._make_desc_list_fuzz(ori_json,
                                                                fuzz_dict,
                                                                ConstManager.INPUT_DESC)
                    output_desc_list = self._make_desc_list_fuzz(ori_json,
                                                                 fuzz_dict,
                                                                 ConstManager.OUTPUT_DESC)
                attr_list = self._check_attr_valid(ori_json, fuzz_dict)
                type_str = output_desc_list[0].get('type')
                if ori_json.get(ConstManager.ST_MODE) == "ms_python_train":
                    suffix_list = ['', type_str]
                    suffix = '_'.join(suffix_list)
                else:
                    format_str = output_desc_list[0].get('format')
                    suffix_list = ['', format_str, type_str]
                    suffix = '_'.join(suffix_list)
                case = {ConstManager.OP: ori_json[ConstManager.OP],
                        ConstManager.INPUT_DESC: input_desc_list,
                        ConstManager.OUTPUT_DESC: output_desc_list,
                        'case_name': prefix + '%03d' % self.case_idx + suffix}
                if ori_json.get(ConstManager.ST_MODE) == "ms_python_train":
                    case[ConstManager.ST_MODE] = "ms_python_train"
                if len(attr_list) > 0:
                    case[ConstManager.ATTR] = attr_list
                self.case_idx, self.total_case_list = \
                    self._add_case_to_total_case(case, self.case_idx,
                                                 [pyfile, function, err_thr],
                                                 self.total_case_list)
        utils.print_info_log('Create %d fuzz test cases for %s.'
                             % (loop_num * len(
            fuzz_return_list) - repeat_case_num,
                                self.json_obj[ConstManager.CASE_NAME]))
        return self.total_case_list

    def check_fuzz_valid(self, json_obj):
        """
        check number match
        :param json_obj: the json_obj of json object
        :return: fuzz_function
        """
        fuzz_impl_path_func = json_obj.get(ConstManager.FUZZ_IMPL)

        if len(fuzz_impl_path_func.split(":")) == 1:
            fuzz_impl_path, fuzz_func = fuzz_impl_path_func, ConstManager.FUZZ_FUNCTION
        elif len(fuzz_impl_path_func.split(":")) == 2:
            fuzz_impl_path, fuzz_func = fuzz_impl_path_func.split(":")
            fuzz_func = fuzz_func.replace(ConstManager.SPACE, ConstManager.EMPTY)
        else:
            utils.print_error_log(
                'The fuzz file "%s" is invalid, ex: fuzz.py:fuzz_function.'
                'Please modify it.' % fuzz_impl_path_func)
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)
        dir_path = os.path.dirname(self.current_json_path)
        real_fuzz_path = os.path.join(dir_path, fuzz_impl_path)
        if os.path.splitext(real_fuzz_path)[-1] != ConstManager.PY_FILE:
            utils.print_error_log(
                'The fuzz file "%s" is invalid, only supports .py file. '
                'Please modify it.' % fuzz_impl_path)
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PATH_ERROR)
        utils.check_path_valid(real_fuzz_path)
        # get fuzz function from fuzz file
        sys.path.append(os.path.dirname(real_fuzz_path))
        fuzz_file = os.path.basename(real_fuzz_path)
        module_name, _ = os.path.splitext(fuzz_file)
        utils.print_info_log("Start to import %s in %s." % (module_name,
                                                            real_fuzz_path))
        try:
            fuzz_function = self._get_fuzz_func(module_name, fuzz_func, real_fuzz_path)
        except Exception as ex:
            utils.print_error_log(
                'Failed to execute function "%s" in %s. %s' % (
                    fuzz_func, fuzz_file, str(ex)))
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR) from ex
        finally:
            pass
        return fuzz_function

    def _check_fuzz_case_num_valid(self, json_obj):
        fuzz_case_num = json_obj.get(ConstManager.FUZZ_CASE_NUM)
        if isinstance(fuzz_case_num, int):
            if 0 < fuzz_case_num <= ConstManager.MAX_FUZZ_CASE_NUM:
                return fuzz_case_num
            utils.print_error_log(
                'The "%s" is invalid in %s, only supports 1~%s. '
                'Please modify it.' % (ConstManager.FUZZ_CASE_NUM,
                                       self.current_json_path,
                                       ConstManager.MAX_FUZZ_CASE_NUM))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
        utils.print_error_log(
            'The "%s" is invalid in %s, only supports integer. '
            'Please modify it.' % (ConstManager.FUZZ_CASE_NUM, self.current_json_path))
        raise utils.OpTestGenException(
            ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)

    def _check_fuzz_value_valid(self, json_tuple, param_type, fuzz_dict,
                                required=True):
        json_obj, key, support_list = json_tuple
        if required:
            self._check_key_exist(json_obj, key, param_type)
        json_obj = self._replace_fuzz_param(json_obj, key, param_type,
                                            fuzz_dict)
        if isinstance(json_obj.get(key), (tuple, list)):
            if len(json_obj.get(key)) != 1:
                utils.print_error_log(
                    'The fuzz case, each of the fields can be configured with '
                    'one profile only. Please modify %s in file %s.' % (key, self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
            param_value = json_obj.get(key)[0]
        else:
            param_value = json_obj.get(key)
        if support_list is not None and param_value not in support_list:
            utils.print_error_log(
                'The value of "%s" for "%s" does not support. '
                'Only supports %s. Please modify it in file %s.' %
                (key, param_type, support_list, self.current_json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        return param_value

    def _check_fuzz_shape_valid(self, json_tuple, param_type, fuzz_dict,
                                required=True):
        json_obj, key = json_tuple
        if required:
            self._check_key_exist(json_obj, key, param_type)
        json_obj = self._replace_fuzz_param(json_obj, key, param_type,
                                            fuzz_dict)
        shape_value = json_obj.get(key)
        if isinstance(shape_value, list):
            self._check_shape_valid(shape_value)
            return shape_value
        utils.print_error_log(
            'The value (%s) is invalid. The key "%s" for "%s" only '
            'supports [] in fuzz case. Please modify it in file %s.' % (
                shape_value, key, param_type, self.current_json_path))
        raise utils.OpTestGenException(
            ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)

    def _check_fuzz_data_distribute_valid(self, json_obj, key, param_type,
                                          fuzz_dict):
        if key in json_obj and ConstManager.VALUE not in json_obj:
            data_distribute_value = self._check_fuzz_value_valid(
                (json_obj, key, self.WHITE_LISTS.data_distribution_list),
                param_type, fuzz_dict)
        else:
            data_distribute_value = 'uniform'
        return data_distribute_value

    def _check_fuzz_value_range_valid(self, json_obj, key, param_type,
                                      fuzz_dict):
        if key in json_obj and ConstManager.VALUE not in json_obj:
            self._check_key_exist(json_obj, key, param_type)
            json_obj = self._replace_fuzz_param(json_obj, key, param_type,
                                                fuzz_dict)
            value_range = json_obj.get(key)
            if isinstance(value_range, list):
                if len(value_range) == 1 and isinstance(value_range[0], list):
                    value_range = value_range[0]
                self._check_range_value_valid(value_range)
            else:
                utils.print_error_log(
                    'The value (%s) is invalid. The key "%s" for "%s" only '
                    'supports [] in fuzz case. Please modify it in file %s.' % (
                        value_range, key, param_type, self.current_json_path))
                raise utils.OpTestGenException(
                    ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        else:
            value_range = [0.1, 1.0]
        return value_range

    def _make_desc_list_ms_fuzz(self, json_obj, fuzz_dict, desc_type):
        desc_list = []
        if len(json_obj[desc_type]) == 0:
            utils.print_error_log(
                'The value of "%s" is empty. Please modify it in '
                'file %s.' % (desc_type, self.current_json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for desc_obj in json_obj[desc_type]:
            type_value = self._check_fuzz_value_valid(
                (desc_obj, 'type', self.WHITE_LISTS.mindspore_type_list),
                desc_type, fuzz_dict)
            shape_value = self._check_fuzz_shape_valid(
                (desc_obj, 'shape'), desc_type, fuzz_dict)
            if desc_type == ConstManager.INPUT_DESC:
                data_distribute = self._check_fuzz_data_distribute_valid(
                    desc_obj, 'data_distribute', desc_type, fuzz_dict)
                value_range = self._check_fuzz_value_range_valid(
                    desc_obj, 'value_range', desc_type, fuzz_dict)
                one_desc = {'type': type_value,
                            'shape': shape_value,
                            'value_range': value_range,
                            'data_distribute': data_distribute}
                # check whether has value.
                desc_obj = self._replace_fuzz_param(
                    desc_obj, ConstManager.VALUE, desc_type, fuzz_dict)
                self._deal_with_value(desc_obj, one_desc, for_fuzz=True)
            else:
                one_desc = {'type': type_value,
                            'shape': shape_value}
            desc_list.append(one_desc)
        return desc_list

    def _deal_with_ori_filed_data_fuzz(self, json_obj, param_type, fuzz_dict,
                                       one_input_desc):
        ori_shape_value = None
        ori_format_value = self._check_fuzz_value_valid(
            (json_obj, 'ori_format', None), param_type, fuzz_dict, required=False)
        if json_obj.get('ori_shape') is not None:
            ori_shape_value = self._check_fuzz_shape_valid(
                (json_obj, 'ori_shape'), param_type, fuzz_dict, required=False)
        if ori_format_value and ori_shape_value:
            one_input_desc['ori_format'] = ori_format_value
            one_input_desc['ori_shape'] = ori_shape_value
        return one_input_desc

    def _check_fuzz_shape_range_valid(self, json_obj, key, param_type,
                                      fuzz_dict):
        json_obj = self._replace_fuzz_param(json_obj, key, param_type,
                                            fuzz_dict)
        shape_range = self._check_list_list_valid(json_obj, key, param_type)
        for item in shape_range:
            self._check_range_value_valid(item, for_shape_range=True)
        return shape_range

    def _deal_with_dynamic_shape_fuzz(self, json_obj, param_type, fuzz_dict,
                                      one_input_desc):
        if not dynamic_handle.check_not_dynamic_shape(json_obj.get('shape')):
            return one_input_desc
        typical_shape = self._check_fuzz_shape_valid(
            (json_obj, ConstManager.TYPICAL_SHAPE), param_type, fuzz_dict)
        dynamic_handle.check_typical_shape_valid(typical_shape,
                                                 self.current_json_path)
        one_input_desc[ConstManager.TYPICAL_SHAPE] = typical_shape
        shape_range = self._check_fuzz_shape_range_valid(
            json_obj, ConstManager.SHAPE_RANGE, param_type, fuzz_dict)
        one_input_desc[ConstManager.SHAPE_RANGE] = shape_range
        return one_input_desc

    def _make_desc_list_fuzz(self, json_obj, fuzz_dict, desc_type):
        desc_list = []
        if len(json_obj[desc_type]) == 0:
            if desc_type == ConstManager.INPUT_DESC:
                utils.print_warn_log(
                    'The value of "input_desc" is empty.')
                return desc_list
            utils.print_error_log(
                'The value of "%s" is empty. Please modify it in '
                'file %s.' % (desc_type, self.current_json_path))
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_INVALID_DATA_ERROR)
        for desc_obj in json_obj[desc_type]:
            format_value = self._check_fuzz_value_valid(
                (desc_obj, 'format', list(self.WHITE_LISTS.format_map.keys())),
                desc_type, fuzz_dict)
            type_value = self._check_fuzz_value_valid(
                (desc_obj, 'type', self.WHITE_LISTS.type_list),
                desc_type, fuzz_dict)
            shape_value = self._check_fuzz_shape_valid(
                (desc_obj, 'shape'), desc_type, fuzz_dict)
            if desc_type == ConstManager.INPUT_DESC:
                data_distribute = self._check_fuzz_data_distribute_valid(
                    desc_obj, 'data_distribute', desc_type, fuzz_dict)
                value_range = self._check_fuzz_value_range_valid(
                    desc_obj, 'value_range', desc_type, fuzz_dict)
                one_desc = {'format': format_value, 'type': type_value,
                            'shape': shape_value,
                            'value_range': value_range,
                            'data_distribute': data_distribute}
                # check whether has value.
                desc_obj = self._replace_fuzz_param(
                    desc_obj, ConstManager.VALUE, desc_type, fuzz_dict)
                self._deal_with_value(desc_obj, one_desc, for_fuzz=True)
            else:
                one_desc = {'format': format_value, 'type': type_value,
                            'shape': shape_value}
            # check whether has ori_format and ori_shape.
            one_desc = self._deal_with_ori_filed_data_fuzz(
                desc_obj, desc_type, fuzz_dict, one_desc)
            # check whether the shape is dynamic.
            if desc_obj.get(ConstManager.TYPICAL_SHAPE) is not None:
                one_desc = self._deal_with_dynamic_shape_fuzz(
                    desc_obj, desc_type, fuzz_dict, one_desc)
            if desc_obj.get("name"):
                one_desc.update({"name": desc_obj.get("name")})
            desc_list.append(one_desc)
        return desc_list

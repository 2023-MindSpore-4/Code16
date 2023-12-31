#!/usr/bin/env python
# coding=utf-8
"""
Function:
AdvanceIniArgs class
This class mainly get the Advance Ini file.
Copyright Information:
Huawei Technologies Co., Ltd. All Rights Reserved © 2021
"""
import os
from io import StringIO
from configparser import RawConfigParser

from op_test_frame.st.interface import utils
from op_test_frame.st.interface.const_manager import ConstManager


class AdvanceIniArgs:
    """
    Class for Load Advance Ini.
    """

    def __init__(self):
        self.only_gen_without_run = 'False'
        self.only_run_without_gen = 'False'
        self.ascend_global_log_level = '3'
        self.ascend_slog_print_to_stdout = '0'
        self.atc_singleop_advance_option = ""
        self.performance_mode = 'False'
        self.compile_options = {}

    def get_ascend_global_log_level(self):
        """
        Function Description: get ascend_global_log_level
        :return: ascend_global_log_level
        """

        return self.ascend_global_log_level

    def get_ascend_slog_print_to_stdout(self):
        """
        Function Description: get ascend_slog_print_to_stdout
        :return: ascend_slog_print_to_stdout
        """

        return self.ascend_slog_print_to_stdout


class AdvanceIniParser:
    """
    Class for Advance Ini Parser.
    """

    def __init__(self, config_file):
        self.config_file = config_file
        self.advance_ini_args = AdvanceIniArgs()
        self.config = RawConfigParser(allow_no_value=True)
        self.advance_args_dic = {
            ConstManager.ONLY_GEN_WITHOUT_RUN: self._init_gen_flag,
            ConstManager.ONLY_RUN_WITHOUT_GEN: self._init_run_flag,
            ConstManager.ASCEND_GLOBAL_LOG_LEVEL: self._init_log_level_env,
            ConstManager.ASCEND_SLOG_PRINT_TO_STDOUT: self._init_slog_flag_env,
            ConstManager.ATC_SINGLEOP_ADVANCE_OPTION: self._init_atc_advance_cmd,
            ConstManager.PERFORMACE_MODE: self._init_performance_mode_flag,
            ConstManager.HOST_ARCH: self._init_host_arch,
            ConstManager.TOOL_CHAIN: self._init_tool_chain
        }

    @staticmethod
    def _check_atc_args_valid(atc_args):
        if len(atc_args) > ConstManager.MAX_NAME_LENGTH:
            utils.print_error_log(
                "The length of value for atc_singleop_advance_option in msopst.ini exceeds %s. "
                "Please modify it." % ConstManager.MAX_NAME_LENGTH)
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
        is_inject_symbol = [False for inject_symbol in ConstManager.INJECT_CHARACTER if inject_symbol in atc_args]
        if is_inject_symbol:
            utils.print_error_log(
                "Configuration error in msopst.ini, the value of "
                "atc_singleop_advance_option = \"%s\" includes inject symbol." % atc_args)
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)

    def get_advance_args_option(self):
        """
        get advance config option.
        """
        try:
            self._read_config_file()
        except utils.OpTestGenException as err:
            utils.print_error_log('Failed to add section to config file')
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_AND_RUN_ERROR) from err
        finally:
            pass
        advance_ini_option_list = self.config.options(
            ConstManager.ADVANCE_SECTION)
        if len(advance_ini_option_list) == 0:
            utils.print_error_log(
                'The %s is empty, please check the file.' % self.config_file)
            raise utils.OpTestGenException(
                ConstManager.OP_TEST_GEN_AND_RUN_ERROR)
        for option in advance_ini_option_list:
            if self.advance_args_dic.get(option):
                self.advance_args_dic.get(option)()
            else:
                utils.print_warn_log(
                    'The %s can not be recognized.' % option)

    def get_mode_flag(self):
        """
        get acl mode flag.
        """
        only_gen = self.advance_ini_args.only_gen_without_run
        only_run = self.advance_ini_args.only_run_without_gen
        performance_mode = self.advance_ini_args.performance_mode
        if only_gen == 'True':
            return ConstManager.ONLY_GEN_WITHOUT_RUN_ACL_PROJ
        if only_run == 'True':
            if performance_mode == 'True':
                return ConstManager.ONLY_RUN_WITHOUT_GEN_ACL_PROJ_PERFORMANCE
            return ConstManager.ONLY_RUN_WITHOUT_GEN_ACL_PROJ
        if performance_mode == 'True':
            return ConstManager.BOTH_GEN_AND_RUN_ACL_PROJ_PERFORMANCE
        return ConstManager.BOTH_GEN_AND_RUN_ACL_PROJ

    def get_env_value(self):
        """
        get env.
        """
        return self.advance_ini_args.ascend_global_log_level, \
               self.advance_ini_args.ascend_slog_print_to_stdout

    def get_atc_advance_cmd(self):
        """
        get atc advance cmd.
        """
        return self.advance_ini_args.atc_singleop_advance_option

    def get_performance_mode_flag(self):
        """
        get performance mode flag.
        """
        if self.advance_ini_args.performance_mode not in ConstManager.TRUE_OR_FALSE_LIST or \
                self.advance_ini_args.performance_mode == \
                ConstManager.TRUE_OR_FALSE_LIST[1]:
            return False
        return True

    def get_compile_options(self):
        """
        get compile options.
        """
        return self.advance_ini_args.compile_options

    def _init_gen_flag(self):
        """
        get value of only_gen_without_run.
        """
        if not self.config.has_option(
                ConstManager.ADVANCE_SECTION,
                ConstManager.ONLY_GEN_WITHOUT_RUN):
            return
        get_gen_flag = self.config.get(
            ConstManager.ADVANCE_SECTION, ConstManager.ONLY_GEN_WITHOUT_RUN)
        if get_gen_flag in ConstManager.TRUE_OR_FALSE_LIST:
            self.advance_ini_args.only_gen_without_run = get_gen_flag
        else:
            utils.print_warn_log(
                'The only_gen_without_run option should be True or False, '
                'please modify it in %s file.' % self.config_file)

    def _init_run_flag(self):
        """
        get value of only_run_without_gen.
        """
        if not self.config.has_option(
                ConstManager.ADVANCE_SECTION,
                ConstManager.ONLY_RUN_WITHOUT_GEN):
            return
        get_run_flag = self.config.get(
            ConstManager.ADVANCE_SECTION, ConstManager.ONLY_RUN_WITHOUT_GEN)
        if get_run_flag in ConstManager.TRUE_OR_FALSE_LIST:
            self.advance_ini_args.only_run_without_gen = get_run_flag
        else:
            utils.print_warn_log(
                'The only_run_without_gen option should be True or False, '
                'please modify it in %s file.' % self.config_file)

    def _init_log_level_env(self):
        """
        get ASCEND_GLOBAL_LOG_LEVEL env.
        """
        if not self.config.has_option(
                ConstManager.ADVANCE_SECTION,
                ConstManager.ASCEND_GLOBAL_LOG_LEVEL):
            return
        get_log_level_env = self.config.get(
            ConstManager.ADVANCE_SECTION, ConstManager.ASCEND_GLOBAL_LOG_LEVEL)
        if get_log_level_env in ConstManager.ASCEND_GLOBAL_LOG_LEVEL_LIST:
            self.advance_ini_args.ascend_global_log_level = get_log_level_env
        else:
            utils.print_warn_log(
                'The ASCEND_GLOBAL_LOG_LEVEL option should be 0-4, '
                'please modify it in %s file.' % self.config_file)

    def _init_slog_flag_env(self):
        """
        get ASCEND_SLOG_PRINT_TO_STDOUT env.
        """
        if not self.config.has_option(
                ConstManager.ADVANCE_SECTION,
                ConstManager.ASCEND_SLOG_PRINT_TO_STDOUT):
            return
        get_slog_flag_env = self.config.get(
            ConstManager.ADVANCE_SECTION,
            ConstManager.ASCEND_SLOG_PRINT_TO_STDOUT)
        if get_slog_flag_env in ConstManager.ASCEND_SLOG_PRINT_TO_STDOUT_LIST:
            self.advance_ini_args.ascend_slog_print_to_stdout = get_slog_flag_env
        else:
            utils.print_warn_log(
                'The ASCEND_SLOG_PRINT_TO_STDOUT option should be 0 or 1, '
                'please modify it in %s file.' % self.config_file)

    def _init_atc_advance_cmd(self):
        """
        get atc advance arguments.
        """
        if not self.config.has_option(
                ConstManager.ADVANCE_SECTION,
                ConstManager.ATC_SINGLEOP_ADVANCE_OPTION):
            return
        get_atc_advance_cmd = self.config.get(
            ConstManager.ADVANCE_SECTION,
            ConstManager.ATC_SINGLEOP_ADVANCE_OPTION)
        atc_advance_args = get_atc_advance_cmd.strip('"')
        self._check_atc_args_valid(atc_advance_args)
        atc_advance_args_list = atc_advance_args.split()
        self.advance_ini_args.atc_singleop_advance_option = atc_advance_args_list

    def _init_host_arch(self):
        """
        get host_arch arguments from config.
        """
        self.advance_ini_args.compile_options[ConstManager.HOST_ARCH] = ""
        if not self.config.has_option(
                ConstManager.ADVANCE_SECTION, ConstManager.HOST_ARCH):
            return
        get_host_arch = self.config.get(
            ConstManager.ADVANCE_SECTION, ConstManager.HOST_ARCH)
        host_arch_args = get_host_arch.strip('"')
        if not host_arch_args:
            return
        if host_arch_args in ConstManager.HOST_ARCH_LIST:
            self.advance_ini_args.compile_options[ConstManager.HOST_ARCH] = host_arch_args
            return
        utils.print_error_log(
            'The HOST_ARCH option only support x86_64 or aarch64, '
            'please modify it in %s file.' % self.config_file)
        raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)

    def _init_tool_chain(self):
        """
        get tool_chain arguments from config.
        """
        self.advance_ini_args.compile_options[ConstManager.TOOL_CHAIN] = ""
        if not self.config.has_option(
                ConstManager.ADVANCE_SECTION, ConstManager.TOOL_CHAIN):
            return
        get_tool_chain = self.config.get(
            ConstManager.ADVANCE_SECTION, ConstManager.TOOL_CHAIN)
        get_tool_chain = get_tool_chain.strip('"')
        if not get_tool_chain:
            return
        try:
            if get_tool_chain.endswith(ConstManager.C_PLUS_PLUS_COMPILER):
                tool_chain_realpath = os.path.realpath(get_tool_chain)
                utils.check_path_valid(tool_chain_realpath, False)
                self.advance_ini_args.compile_options[
                    ConstManager.TOOL_CHAIN] = tool_chain_realpath
            else:
                utils.print_error_log(
                    'The path of TOOL_CHAIN should be end with g++, '
                    'the actual value is %s.' % get_tool_chain)
                raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR)
        except utils.OpTestGenException as err:
            utils.print_error_log('The path of TOOL_CHAIN is invalid, '
                                  ' Please check and modify it in %s file.' % self.config_file)
            raise utils.OpTestGenException(ConstManager.OP_TEST_GEN_INVALID_PARAM_ERROR) from err

    def _init_performance_mode_flag(self):
        """
        get value of performance_mode.
        """
        if not self.config.has_option(
                ConstManager.ADVANCE_SECTION, ConstManager.PERFORMACE_MODE):
            return
        get_performance_mode_flag = self.config.get(
            ConstManager.ADVANCE_SECTION, ConstManager.PERFORMACE_MODE)
        if get_performance_mode_flag in ConstManager.TRUE_OR_FALSE_LIST:
            self.advance_ini_args.performance_mode = get_performance_mode_flag
        else:
            utils.print_warn_log(
                'The performance_mode option should be True or False, '
                'please modify it in %s file.' % self.config_file)

    def _read_config_file(self):
        with open(self.config_file, encoding='UTF-8') as msopst_conf_file:
            conf_file_context = msopst_conf_file.read()
        with StringIO('[RUN]\n%s' % conf_file_context) as section_file:
            self.config.read_file(section_file)

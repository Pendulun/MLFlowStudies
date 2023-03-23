import unittest

import argparse
from model_exec_helper import BaseArgumentParser, NoTestPathError, NoTrainPathError

class BaseArgumentParserTest(unittest.TestCase):

    def test_get_train_arg_without_parent_parser(self):
        bap = BaseArgumentParser()

        train_path = "."
        arguments = ("--train --train-path "+train_path).split()
        parsed_args = bap.parse_args(entry_args=arguments)

        self.assertTrue(parsed_args.train)
        self.assertEqual(parsed_args.train_path, train_path)
    
    def test_get_test_arg_without_parent_parser(self):
        bap = BaseArgumentParser()

        test_path = "."
        arguments = ("--test --test-path "+test_path).split()
        parsed_args = bap.parse_args(entry_args=arguments)

        self.assertTrue(parsed_args.test)
        self.assertEqual(parsed_args.test_path, test_path)
    
    def test_get_train_and_test_args_without_parent_parser(self):
        bap = BaseArgumentParser()

        test_path = "."
        train_path = "."
        arguments = ("--test --test-path "+test_path+" --train --train-path "+train_path).split()
        parsed_args = bap.parse_args(entry_args=arguments)

        self.assertTrue(parsed_args.test)
        self.assertEqual(parsed_args.test_path, test_path)

        self.assertTrue(parsed_args.train)
        self.assertEqual(parsed_args.train_path, train_path)
    
    def test_get_custom_arg_from_parent_parser(self):
        parent_parser = argparse.ArgumentParser()
        parent_parser.add_argument("--custom-arg", action='store_true')

        bap = BaseArgumentParser(parent_parser)

        arguments = ("--custom-arg ").split()
        parsed_args = bap.parse_args(entry_args=arguments)

        self.assertTrue(parsed_args.custom_arg)
    
    def test_get_custom_and_base_args(self):
        parent_parser = argparse.ArgumentParser()
        parent_parser.add_argument("--custom-arg", action='store_true')

        bap = BaseArgumentParser(parent_parser)

        test_path = "."
        train_path = "."
        arguments = ("--custom-arg --test --test-path "+test_path+" --train --train-path "+train_path).split()
        parsed_args = bap.parse_args(entry_args=arguments)

        self.assertTrue(parsed_args.custom_arg)

        self.assertTrue(parsed_args.test)
        self.assertEqual(parsed_args.test_path, test_path)

        self.assertTrue(parsed_args.train)
        self.assertEqual(parsed_args.train_path, train_path)

if __name__ == "__main__":
    unittest.main()
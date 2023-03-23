import argparse
import pathlib

class WriteParserError(Exception):
    pass

class NoTrainPathError(Exception):
    pass

class NoTestPathError(Exception):
    pass

class BaseArgumentParser():
    
    def __init__(self, parser:argparse.ArgumentParser = None):
        """
        This is a container to an argparse.ArgumentParser with base arguments to be used along all models
        to be trained/tested.

        The base arguments are: 
        --train: If it should train a model
        --train-path: The path to the training instances. This must be a file or a dict path.
        --test: If it should test the model
        --test-path: The path to the testing instances. This must be a file or a dict path.

        parser: ArgumentParser that will be the parent of this BaseArgumentParser. This
            BaseArgumentParser will copy its prog, description and epilog attributes.
            See https://docs.python.org/3/library/argparse.html#parents for more.
        """
        
        self._parent_parser = parser

        parent_info = self._get_ref_parser_info(self._parent_parser)
        parents = [parser] if self._parent_parser else []

        self._final_parser = argparse.ArgumentParser(**parent_info, parents=parents,
                                                     conflict_handler='resolve')
        
        self._final_parser = self._config_parser(self._final_parser)

    
    def _get_ref_parser_info(self, parser: argparse.ArgumentParser) -> dict:
        """
        Get some target attributes from the parser and returns them in a dict.
        """
        return_dict = {'prog':None, 'description':None, 'epilog':None}
        if parser:
            return_dict['prog'] = parser.prog
            return_dict['description'] = parser.description
            return_dict['epilog'] = parser.epilog
        
        return return_dict
    
    def _config_parser(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Configure the base parser to be contained 
        """
        parser = self._add_base_args(parser)
        return parser

    def _add_base_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add base arguments that should be accepted by the parser and returns it 
        """
        parser.add_argument("--train", required=False, action='store_true',
                            help="Indicates that the model should be trained")
        parser.add_argument("--test", required=False, action='store_true',
                            help="Indicates that the model should be tested")
        parser.add_argument("--train-path", required=False,
                            help="The path to a single file or a dir with files.")
        parser.add_argument("--test-path", required=False, 
                            help="The path to a single file or a dir with files.")

        return parser
    
    @property
    def parser(self):
        return self._final_parser

    @parser.setter
    def parser(self):
        raise WriteParserError("The parser is read only")
    
    def parse_args(self, entry_args:list = None):
        """
        Parse arguments from the command line or from entry_args if provided
        """
        args = self._final_parser.parse_args(entry_args)

        if args.train and not args.train_path:
            raise NoTrainPathError("'train' argument used but no 'train-path' provided!")
        
        if args.test and not args.test_path:
            raise NoTrainPathError("'test' argument used but no 'test-path' provided!")
        
        return args
from logging.config import dictConfig
from logging import getLogger as _getLogger


config = {
    'version': 1,
    'formatters': {
        'detailed': {
            'class': 'logging.Formatter',
            'format': '%(asctime)s %(name)-15s %(levelname)-8s %(processName)-10s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'DEBUG',
        },
    },
    'loggers': {
        '': {
            'level': 'INFO',
            'handlers': ['console']
        },
        'mlmi': {
            'level': 'DEBUG',
            'propagate': False,
            'handlers': ['console']
        },
        '__main__': {
            'level': 'DEBUG',
            'propagate': False,
            'handlers': ['console']
        }
    }
}


dictConfig(config)
getLogger = _getLogger

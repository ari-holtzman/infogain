# Models

## OpenAI

GPT2_S = 'gpt2'
GPT2_M = 'gpt2-medium'
GPT2_L = 'gpt2-large'
GPT2_X = 'gpt2-xl'

GPT2 = [ GPT2_S, GPT2_M, GPT2_L, GPT2_X ]

GPT3_S = 'ada'
GPT3_M = 'babbage'
GPT3_L = 'curie'
GPT3_X = 'davinci'

GPT3 = [ GPT3_S, GPT2_M, GPT3_L, GPT3_X ]

CODEX_L = 'cushman-codex'
CODEX_X = 'davinci-codex'

CODEX = [ CODEX_L, CODEX_X ]

OPENAI_API = [ GPT3_S, GPT3_M, GPT3_L, GPT3_X, CODEX_L, CODEX_X ]

# Eleuther 

GPTN_125M  = 'EleutherAI/gpt-neo-125M' 
GPTN_1300M = 'EleutherAI/gpt-neo-1.3B'
GPTN_2700M = 'EleutherAI/gpt-neo-2.7B'

GPTJ = 'EleutherAI/gpt-j-6B'

ELEUTHER = [ GPTN_125M, GPTN_1300M, GPTN_2700M, GPTJ ]


# Google
# TODO

# Facebook

## OPT

OPT_125M = 'facebook/opt-125m'
OPT_350M = 'facebook/opt-350m'
OPT_1P3B = 'facebook/opt-1.3b'
OPT_2P7B = 'facebook/opt-2.7b'
OPT_6P7B = 'facebook/opt-6.7b'
OPT_13B  = 'facebook/opt-13b'
OPT_30B  = 'facebook/opt-30b'
OPT_66B  = 'facebook/opt-66b'

OPT = [ OPT_125M, OPT_350M, OPT_1P3B, OPT_2P7B, OPT_6P7B, OPT_13B, OPT_30B, OPT_66B ]

model = { 
            '2s'    : GPT2_S, 
            '2m'    : GPT2_M, 
            '2l'    : GPT2_L, 
            '2x'    : GPT2_X,
            '3s'    : GPT3_S, 
            '3m'    : GPT3_M, 
            '3l'    : GPT3_L, 
            '3x'    : GPT3_X,
            'n125m' : GPTN_125M,
            'n1p3b' : GPTN_1300M,
            'n2p7b' : GPTN_2700M, 
            'j'     : GPTJ,
            'opt-125m' : OPT_125M,
            'opt-350m' : OPT_350M,
            'opt-1.3b' : OPT_1P3B,
            'opt-2.7b' : OPT_2P7B,
            'opt-6.7b' : OPT_6P7B,
            'opt-13b'  : OPT_13B,
            'opt-30b'  : OPT_30B,
            'opt-66b'  : OPT_66B,
         }


cw_length = {
                GPT2_S     : 1024,
                GPT2_M     : 1024,
                GPT2_L     : 1024,
                GPT2_X     : 1024,
                GPT3_S     : 2048,
                GPT3_M     : 2048,
                GPT3_L     : 2048,
                GPT3_X     : 2048,
                CODEX_L    : 2048,
                CODEX_X    : 2048, 
                GPTN_125M  : 2048,
                GPTN_1300M : 2048,
                GPTN_2700M : 2048,
                GPTJ       : 2048,
                OPT_125M   : 2048, 
                OPT_350M   : 2048,    
                OPT_1P3B   : 2048,    
                OPT_2P7B   : 2048,    
                OPT_6P7B   : 2048,    
                OPT_13B    : 2048,    
                OPT_30B    : 2048,    
                OPT_66B    : 2048,   
}

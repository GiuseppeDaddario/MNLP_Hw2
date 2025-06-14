#!/bin/bash
eval $(ssh-agent)
step ssh login 'daddario.2177530@studenti.uniroma1.it' --provisioner cineca-hpc
ssh gdaddari@login.leonardo.cineca.it
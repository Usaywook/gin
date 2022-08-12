#!/usr/bin/env bash
docker kill $(docker ps -q)
kill -9 $(lsof -i TCP:8000)
kill -9 $(lsof -i TCP:8050)
kill -9 $(lsof -i TCP:8100)
kill -9 $(lsof -i TCP:2000)
kill -9 $(lsof -i TCP:2002)
kill -9 $(lsof -i TCP:2004)
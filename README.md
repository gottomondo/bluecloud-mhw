# bluecloud-mhw
test of code integration with ccp

## token
```
TOKEN=$(curl -X POST $ccpiamurl -d grant_type=refresh_token -d client_id=$ccpclientid -d refresh_token=$ccprefreshtoken -H "X-D4Science-Context: $ccpcontext" | jq -r '."access_token"')
curl {{filews}} -o {{localinputfile}} -H "Authorization: Bearer $TOKEN"
```

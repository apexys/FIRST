
function corpus(name){
	var result = "";
	for(let i = 0; i < 10000; i++){
		result += `corpus-${name}/${i}.png ${i}\n`;
	}
	require('fs').writeFileSync(`corpus-${name}.txt`, result, 'utf-8');
}

corpus("da");
corpus("wo");
corpus("was");
corpus("wie");
let contador_padrao = 0;

function clicar_botao() {
    contador_padrao = contador_padrao + 1;
    document.getElementById('contador_valor').textContent = contador_padrao;
    console.log(contador_padrao);
}

document.querySelector('.meu_botao').addEventListener('click', clicar_botao);

// AViS CGDF — bundle gerado automaticamente (sem módulos ES)
// Funciona via file:// e via HTTP server

"use strict";

// ── js/rag/tokenizer.js ──
/**
 * tokenizer.js — Normalização e tokenização para português (BR).
 *
 * Responsável por transformar texto livre em uma lista de tokens comparáveis:
 * remove acentos, coloca em minúsculas, separa por não-letras e descarta
 * stopwords (palavras muito frequentes que não ajudam na recuperação).
 *
 * Usado tanto na indexação da base de conhecimento quanto nas consultas do
 * usuário, garantindo que "Férias", "ferias" e "FÉRIAS" caiam no mesmo token.
 */

// Faixa Unicode dos sinais diacríticos combinantes (acentos decompostos via NFD).
const DIACRITICOS = /[̀-ͯ]/g;

// Stopwords comuns do português + termos genéricos do domínio (SEI/processo)
// que aparecem em quase todos os POPs e, por isso, não discriminam documentos.
const STOPWORDS = new Set([
  'a', 'o', 'as', 'os', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos',
  'das', 'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com', 'sem', 'sob',
  'sobre', 'e', 'ou', 'mas', 'que', 'se', 'ao', 'aos', 'a', 'as', 'como',
  'quando', 'onde', 'qual', 'quais', 'meu', 'minha', 'meus', 'minhas', 'seu',
  'sua', 'seus', 'suas', 'este', 'esta', 'esse', 'essa', 'isso', 'isto', 'ser',
  'estar', 'ter', 'haver', 'fazer', 'poder', 'dever', 'me', 'te', 'lhe',
  'vos', 'eu', 'tu', 'ele', 'ela', 'nos', 'vos', 'eles', 'elas', 'ja',
  'nao', 'sim', 'mais', 'menos', 'muito', 'pouco', 'todo', 'toda',
  'gostaria', 'quero', 'preciso', 'saber', 'sei', 'segundo', 'conforme',
]);

/**
 * Remove acentuação preservando a letra base (á -> a, ç -> c).
 * @param {string} texto
 * @returns {string}
 */
function removerAcentos(texto) {
  return String(texto).normalize('NFD').replace(DIACRITICOS, '');
}

/**
 * Normaliza uma string: minúsculas + sem acentos.
 * @param {string} texto
 * @returns {string}
 */
function normalizar(texto) {
  return removerAcentos(String(texto).toLowerCase());
}

/**
 * Tokeniza um texto em termos relevantes para recuperação.
 * @param {string} texto
 * @param {{ manterStopwords?: boolean, tamanhoMinimo?: number }} [opcoes]
 * @returns {string[]}
 */
function tokenizar(texto, opcoes = {}) {
  const { manterStopwords = false, tamanhoMinimo = 2 } = opcoes;
  const normalizado = normalizar(texto);
  const brutos = normalizado.split(/[^a-z0-9]+/);
  return brutos.filter((token) => {
    if (token.length < tamanhoMinimo) return false;
    if (!manterStopwords && STOPWORDS.has(token)) return false;
    return true;
  });
}

// ── js/rag/engine.js ──
/**
 * engine.js — Motor de recuperação (RAG) client-side com ranqueamento BM25.
 *
 * Este é o núcleo técnico do protótipo: em vez de casar palavras-chave com
 * `if/else`, indexamos passagens REAIS dos POPs e da LC 840 e, a cada consulta,
 * pontuamos cada passagem com o algoritmo BM25 (o mesmo usado por motores de
 * busca como Elasticsearch/Lucene). O resultado é uma recuperação genuína,
 * tolerante a sinônimos e ordem das palavras, com trecho-fonte e score.
 *
 * BM25 (Okapi): para um termo t e documento d,
 *   score = IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d|/avgdl))
 * onde IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)).
 */

const K1 = 1.5; // saturação de frequência do termo
const B = 0.75; // normalização pelo comprimento do documento

class RagEngine {
  /**
   * @param {import('../data/knowledge-base.js').Documento[]} documentos
   */
  constructor(documentos) {
    this.documentos = documentos;
    /** @type {Array<{docId:string, chunkIndex:number, texto:string, tokens:string[], tf:Map<string,number>, tamanho:number}>} */
    this.passagens = [];
    /** @type {Map<string, number>} termo -> nº de passagens que o contêm */
    this.docFreq = new Map();
    this.tamanhoMedio = 0;
    this._indexar();
  }

  /** Constrói o índice invertido/estatísticas a partir das passagens reais. */
  _indexar() {
    for (const doc of this.documentos) {
      // Reforço de recuperação: sinônimos/palavras-chave do domínio entram no
      // texto indexável do 1º chunk, sem poluir a resposta exibida ao usuário.
      const chunks = doc.chunks || [];
      chunks.forEach((texto, chunkIndex) => {
        const reforco = chunkIndex === 0 && doc.keywords ? ' ' + doc.keywords.join(' ') : '';
        const tokens = tokenizar(texto + reforco);
        const tf = new Map();
        for (const tok of tokens) tf.set(tok, (tf.get(tok) || 0) + 1);
        for (const termo of tf.keys()) {
          this.docFreq.set(termo, (this.docFreq.get(termo) || 0) + 1);
        }
        this.passagens.push({
          docId: doc.id,
          chunkIndex,
          texto,
          tokens,
          tf,
          tamanho: tokens.length,
        });
      });
    }
    const total = this.passagens.reduce((s, p) => s + p.tamanho, 0);
    this.tamanhoMedio = this.passagens.length ? total / this.passagens.length : 0;
  }

  /**
   * IDF suavizado (BM25) de um termo.
   * @param {string} termo
   * @returns {number}
   */
  _idf(termo) {
    const N = this.passagens.length;
    const n = this.docFreq.get(termo) || 0;
    return Math.log(1 + (N - n + 0.5) / (n + 0.5));
  }

  /**
   * Pontua uma passagem contra os tokens da consulta usando BM25.
   * @param {{tf:Map<string,number>, tamanho:number}} passagem
   * @param {string[]} tokensConsulta
   * @returns {number}
   */
  _scoreBM25(passagem, tokensConsulta) {
    let score = 0;
    for (const termo of tokensConsulta) {
      const f = passagem.tf.get(termo);
      if (!f) continue;
      const idf = this._idf(termo);
      const numerador = f * (K1 + 1);
      const denominador =
        f + K1 * (1 - B + B * (passagem.tamanho / (this.tamanhoMedio || 1)));
      score += idf * (numerador / denominador);
    }
    return score;
  }

  /**
   * Consulta a base e retorna passagens ranqueadas + documento vencedor.
   *
   * @param {string} textoConsulta
   * @param {{ topK?: number }} [opcoes]
   * @returns {{
   *   documento: import('../data/knowledge-base.js').Documento | null,
   *   confianca: number,
   *   melhorTrecho: string,
   *   fontes: Array<{docId:string, titulo:string, arquivoFonte:string, score:number, trecho:string}>,
   *   tokensConsulta: string[]
   * }}
   */
  query(textoConsulta, opcoes = {}) {
    const { topK = 3 } = opcoes;
    const tokensConsulta = tokenizar(textoConsulta);

    const ranqueadas = this.passagens
      .map((p) => ({ passagem: p, score: this._scoreBM25(p, tokensConsulta) }))
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score);

    if (ranqueadas.length === 0) {
      return {
        documento: null,
        confianca: 0,
        melhorTrecho: '',
        fontes: [],
        tokensConsulta,
      };
    }

    // Agrega score por documento para eleger o POP mais relevante no todo.
    const scorePorDoc = new Map();
    for (const r of ranqueadas) {
      scorePorDoc.set(
        r.passagem.docId,
        (scorePorDoc.get(r.passagem.docId) || 0) + r.score,
      );
    }
    const docIdVencedor = [...scorePorDoc.entries()].sort((a, b) => b[1] - a[1])[0][0];
    const documento = this.documentos.find((d) => d.id === docIdVencedor) || null;

    const fontes = ranqueadas.slice(0, topK).map((r) => {
      const doc = this.documentos.find((d) => d.id === r.passagem.docId);
      return {
        docId: r.passagem.docId,
        titulo: doc ? doc.titulo : r.passagem.docId,
        arquivoFonte: doc ? doc.arquivoFonte : '',
        score: r.score,
        trecho: r.passagem.texto,
      };
    });

    // Confiança normalizada (0–1): satura o melhor score bruto para uma escala
    // legível ao usuário. Heurística de apresentação, não probabilidade formal.
    const melhorScore = ranqueadas[0].score;
    const confianca = Math.max(0, Math.min(1, melhorScore / (melhorScore + 3)));

    return {
      documento,
      confianca,
      melhorTrecho: ranqueadas[0].passagem.texto,
      fontes,
      tokensConsulta,
    };
  }
}

// ── js/data/knowledge-base.js ──
/**
 * knowledge-base.js — Base de conhecimento estruturada do AViS.
 *
 * Cada documento representa um Procedimento Operacional Padrão (POP) oficial da
 * CGDF. Os campos `chunks` contêm PASSAGENS REAIS extraídas dos PDFs em `pop/`
 * (é sobre esse texto que o motor BM25 pontua a recuperação). Os demais campos
 * (`passosSEI`, `baseLegal`, etc.) estruturam a resposta exibida ao usuário.
 *
 * Fonte dos textos: PDFs oficiais da pasta `pop/` deste repositório.
 *
 * @typedef {Object} PassoSEI
 * @property {string} titulo
 * @property {string[]} itens
 *
 * @typedef {Object} Documento
 * @property {string} id
 * @property {string} titulo
 * @property {string} icone           - classe FontAwesome
 * @property {string} categoria
 * @property {string} arquivoFonte    - nome do PDF de origem (para citação)
 * @property {string} versao
 * @property {string} baseLegal
 * @property {string} unidadeDestino
 * @property {string[]} keywords      - sinônimos p/ reforço de recuperação
 * @property {string[]} chunks        - passagens reais indexadas pelo BM25
 * @property {string} [prazos]        - resumo de prazos (opcional)
 * @property {PassoSEI[]} passosSEI
 * @property {string} [dica]
 */

/** @type {Documento[]} */
const BASE_CONHECIMENTO = [
  {
    id: 'ferias',
    titulo: 'Marcação, Remarcação e Suspensão de Férias',
    icone: 'fa-umbrella-beach',
    categoria: 'POP — Férias',
    arquivoFonte: 'POP-Ferias-2025-1.pdf',
    versao: 'Versão 07 — 15/04/2025',
    baseLegal: 'LC nº 840/2011, IN nº 1/2024 e Memorando Circular nº 2/2025 - CGDF/SUBGI',
    unidadeDestino: 'CGDF/SUBGI/COGEP/DITEC/GEREF',
    keywords: [
      'ferias', 'marcar', 'remarcar', 'remarcacao', 'suspender', 'suspensao',
      'agendar', 'descanso', 'recesso', 'gozar', 'usufruir', 'periodo aquisitivo',
    ],
    prazos: '1º período: 60 dias de antecedência • 2º e 3º períodos: 15 dias de antecedência',
    chunks: [
      'Processo Pessoal: Marcação de Férias. 1º período: 60 dias de antecedência. 2º e 3º período: 15 dias de antecedência. Iniciar (abrir ou reabrir) o tipo do processo Pessoal: Férias. No campo especificação e interessado, preencher seu nome completo, em caixa alta. Utilizar 1 (um) único processo para todos os exercícios.',
      'Marcação de férias: incluir o tipo do documento Requerimento - Férias: Marcação (Formulário). Preencher e assinar o formulário e solicitar a assinatura da sua chefia imediata ou mediata. Confirmar as assinaturas e encaminhar o processo à CGDF/SUBGI/COGEP/DITEC/GEREF.',
      'Processo Pessoal: Remarcação de Férias. 1º período: 60 dias de antecedência. 2º e 3º período: 15 dias de antecedência. Reabrir o processo Pessoal: Férias e consultar o número do processo no "Meu perfil" da intranet. Incluir o documento Requerimento - Férias: Remarcação (Formulário), preencher, assinar e obter a assinatura da chefia.',
      'Processo Pessoal: Suspensão de Férias. Usufruir no mínimo 1 dia. Motivos de suspensão: calamidade pública, comoção interna, convocação para júri, serviço militar ou eleitoral, ou por necessidade do serviço. Incluir o documento Memorando informando a motivação legal, a data da suspensão e a data de usufruto do saldo remanescente, seguindo a ordem cronológica.',
      'Suspensão de férias: o chefe imediato ou mediato assina o Memorando e solicita autorização do Controlador-Geral Adjunto, que também assina. Encaminhar o processo para CGDF/SUBGI/COGEP/DITEC/GEREF antes da suspensão. Após lançamento a COGEP insere a ciência. Verificar se há formulário de devolução, corrigir e reenviar à GEREF/COGEP.',
    ],
    passosSEI: [
      {
        titulo: '1. Abrir/Reabrir o processo',
        itens: [
          'Tipo: <code>Pessoal: Férias</code> — use <strong>1 único processo</strong> para todos os exercícios.',
          'Especificação e Interessado: seu nome completo em <strong>CAIXA ALTA</strong>.',
        ],
      },
      {
        titulo: '2. Incluir o formulário',
        itens: [
          'Marcação: <em>Requerimento - Férias: Marcação (Formulário)</em>.',
          'Remarcação: <em>Requerimento - Férias: Remarcação (Formulário)</em>.',
          'Suspensão: <em>Memorando</em> com motivação legal + autorização do Controlador-Geral Adjunto (usufruir no mínimo 1 dia).',
          'Preencher, assinar e obter assinatura da chefia imediata ou mediata.',
        ],
      },
      {
        titulo: '3. Encaminhar',
        itens: [
          'Conferir as assinaturas e enviar para <code>CGDF/SUBGI/COGEP/DITEC/GEREF</code>.',
          'Servidor cedido/requisitado: encaminhar também ao órgão de origem.',
        ],
      },
    ],
    dica: 'Acompanhe o pedido na aba "Férias" do "Meu perfil" da intranet. Se houver formulário de devolução, corrija e reenvie à GEREF/COGEP.',
  },

  {
    id: 'abono_5dias',
    titulo: 'Abono de Ponto Anual (5 dias)',
    icone: 'fa-calendar-minus',
    categoria: 'POP — Abono de Ponto',
    arquivoFonte: 'POP-Abono-de-Ponto.pdf',
    versao: 'Versão 3.0 — 22/04/2019',
    baseLegal: 'LC nº 840/2011',
    unidadeDestino: 'CGDF/SUBGI/COGEP/DITEC/GEREF',
    keywords: [
      'abono', 'ponto', 'folga', 'cinco dias', '5 dias', 'anual', 'faltar',
      'dia de folga', 'abonar',
    ],
    chunks: [
      'Processo Pessoal: Abono de Ponto. Iniciar o tipo do processo Pessoal: Abono de Ponto. No campo Especificação, preencha seu nome completo em caixa alta. O nível de acesso deve ser Público. Para cada exercício, um novo processo de abono de ponto.',
      'Abono de ponto: incluir o tipo do documento Requerimento – Abono de Ponto (Formulário). Preencher os campos e confirmar os dados. Solicitar a assinatura da chefia imediata; se a chefia for de outra Unidade, inserir o documento em bloco de assinatura e disponibilizar o bloco.',
      'Abono de ponto encaminhamento: verificar a assinatura do interessado e da chefia imediata e encaminhar o processo para CGDF/SUBGI/COGEP/DITEC/GEREF. Quando a GEREF faz o lançamento, inclui a ciência no Requerimento. As marcações de abono ficam disponíveis no "perfil" da Intranet para consulta.',
      'Cancelamento de abono de ponto: o servidor deverá reabrir o processo, inserir um novo formulário, preencher os dados pessoais e o campo reservado ao cancelamento; depois seguir a assinatura da chefia e o encaminhamento à GEREF.',
    ],
    passosSEI: [
      {
        titulo: '1. Abrir o processo',
        itens: [
          'Tipo: <code>Pessoal: Abono de Ponto</code> (um novo processo por exercício).',
          'Especificação: nome completo em <strong>CAIXA ALTA</strong>. Nível de acesso: <strong>Público</strong>.',
        ],
      },
      {
        titulo: '2. Incluir o formulário',
        itens: [
          'Documento: <em>Requerimento – Abono de Ponto (Formulário)</em>.',
          'Chefia da mesma Unidade: peça a assinatura direta. De outra Unidade: use <strong>bloco de assinatura</strong>.',
        ],
      },
      {
        titulo: '3. Encaminhar',
        itens: ['Após as assinaturas, enviar para <code>CGDF/SUBGI/COGEP/DITEC/GEREF</code>.'],
      },
    ],
    dica: 'Para cancelar, reabra o processo, inclua novo formulário preenchendo o campo de cancelamento e repita as etapas de assinatura e envio.',
  },

  {
    id: 'abono_aniversario',
    titulo: 'Abono de Ponto — Dia do Aniversário',
    icone: 'fa-cake-candles',
    categoria: 'POP — Abono de Ponto',
    arquivoFonte: 'POP-Abono-de-ponto-Dia-do-aniversario.pdf',
    versao: 'Versão 1.0 — 24/12/2025',
    baseLegal: 'Lei nº 7.826, de 18/12/2025',
    unidadeDestino: 'CGDF/SUBGI/COGEP/DITEC/GEREF (e órgão de origem)',
    keywords: [
      'aniversario', 'niver', 'data de nascimento', 'dia do aniversario',
      '7826', 'lei 7826', 'um dia', 'abono aniversario',
    ],
    chunks: [
      'Processo Pessoal: Abono de ponto – Dia do Aniversário. Iniciar ou reabrir o tipo do processo Pessoal: Abono de ponto. No campo especificação e interessado, preencher seu nome completo em caixa alta. O nível de acesso é público. Utilize o mesmo processo de abono de ponto de 5 dias.',
      'Abono aniversário: incluir o tipo do documento Requerimento Geral. Editar e assinar o documento inserindo a solicitação da data de usufruto e a informação da data do seu aniversário. Solicitar a assinatura da chefia imediata ou mediata.',
      'Abono aniversário encaminhamento: verificar se o documento está assinado e encaminhar o processo para CGDF/SUBGI/COGEP/DITEC/GEREF e para o órgão de origem.',
      'A Lei 7.826/2025 concede ao servidor público do Distrito Federal o direito a 1 dia de abono de ponto por ano, no dia do seu aniversário, sem prejuízo dos vencimentos. A concessão é na data do aniversário, não podendo ser antecipada ou postergada fora das exceções. Há regras específicas para perda do benefício; se ocorrer a situação do Art. 4º, a chefia deverá justificar para o processo não ser devolvido.',
    ],
    passosSEI: [
      {
        titulo: '1. Usar o processo de abono do exercício',
        itens: [
          'Tipo: <code>Pessoal: Abono de ponto</code> — <strong>o mesmo</strong> processo do abono de 5 dias.',
          'Especificação: nome completo em CAIXA ALTA. Nível: Público.',
        ],
      },
      {
        titulo: '2. Incluir o Requerimento Geral',
        itens: [
          'Documento: <em>Requerimento Geral</em>.',
          'Informar a <strong>data do aniversário</strong> e a data de usufruto solicitada; assinar e obter assinatura da chefia.',
        ],
      },
      {
        titulo: '3. Encaminhar',
        itens: ['Enviar para <code>CGDF/SUBGI/COGEP/DITEC/GEREF</code> e para o órgão de origem.'],
      },
    ],
    dica: 'A concessão é no dia exato do aniversário (Lei 7.826/2025) — sem antecipar nem postergar, salvo exceções. Atenção às regras de perda do benefício.',
  },

  {
    id: 'auxilio_alimentacao',
    titulo: 'Auxílio Alimentação / Refeição',
    icone: 'fa-utensils',
    categoria: 'POP — Benefícios',
    arquivoFonte: 'POP-Auxilio-alimentacao.pdf',
    versao: 'Versão 3.0 — 06/04/2020',
    baseLegal: 'LC nº 840/2011',
    unidadeDestino: 'CGDF/SUBGI/COGEP/DITEC/GERFI',
    keywords: [
      'auxilio', 'alimentacao', 'refeicao', 'vale', 'ticket', 'opcao',
      'auxilio alimentacao', 'auxilio refeicao', 'beneficio alimentacao',
    ],
    chunks: [
      'Processo Pessoal: Auxílio Alimentação/Refeição. Iniciar o tipo do processo Pessoal: Auxílio Alimentação/Refeição. No campo Especificação e Interessado, preencher seu nome completo em caixa alta. O nível de acesso deve ser Público.',
      'Auxílio alimentação: incluir o tipo do documento "Termo de Opção de Auxílio-Alimentação – Formulário". Preencher os campos, assinar e encaminhar à CGDF/SUBGI/COGEP/DITEC/GERFI. Base legal: LC 840/2011.',
    ],
    passosSEI: [
      {
        titulo: '1. Abrir o processo',
        itens: [
          'Tipo: <code>Pessoal: Auxílio Alimentação/Refeição</code>.',
          'Especificação e Interessado: nome completo em CAIXA ALTA. Nível: Público.',
        ],
      },
      {
        titulo: '2. Incluir o Termo de Opção',
        itens: ['Documento: <em>Termo de Opção de Auxílio-Alimentação – Formulário</em>. Preencher e assinar.'],
      },
      {
        titulo: '3. Encaminhar',
        itens: ['Enviar para <code>CGDF/SUBGI/COGEP/DITEC/GERFI</code>.'],
      },
    ],
    dica: 'Diferente dos POPs de ponto/férias, o destino é a <strong>GERFI</strong> (não a GEREF).',
  },

  {
    id: 'pos_graduacao',
    titulo: 'Capacitação em Serviço — Pós-Graduação Stricto Sensu',
    icone: 'fa-graduation-cap',
    categoria: 'POP — Capacitação',
    arquivoFonte: 'POP-Capacitacao-em-Servico-02.02.pdf',
    versao: 'Referência 01/2026',
    baseLegal: 'Portaria CGDF nº 32, de 22/01/2026',
    unidadeDestino: 'CGDF/SUBGI/COGEP/DIEST',
    keywords: [
      'capacitacao', 'pos graduacao', 'pos-graduacao', 'mestrado', 'doutorado',
      'pos-doutorado', 'stricto sensu', 'reduzir jornada', 'reducao de jornada',
      'estudar', 'carga horaria', 'portaria 32', 'liberacao para estudar',
    ],
    prazos: 'Mestrado: até 24 meses (3 anos de CGDF) • Doutorado: até 48 meses • Pós-doutorado: até 12 meses (4 anos de CGDF)',
    chunks: [
      'Capacitação em serviço em programa de pós-graduação stricto sensu (Portaria CGDF nº 32, de 22 de janeiro de 2026). Faculta-se ao servidor a capacitação em serviço quando não puder ser realizada a compensação de horas na jornada semanal regular ou quando não houver possibilidade de afastamento integral por necessidade de serviço.',
      'O servidor poderá capacitar-se em serviço por até 30 por cento da carga horária semanal da sua carreira, mediante autorização da chefia imediata, chefia mediata e Subcontrolador, Ouvidor-Geral ou Controlador-Geral Adjunto, sem necessidade de compensação de horário, por período máximo de 24 meses para mestrado, 48 meses para doutorado e 12 meses para pós-doutorado.',
      'Requisitos: ser ocupante de cargo efetivo do GDF e estar em exercício na CGDF ou nas Unidades de Controle Interno há 3 anos consecutivos para mestrado e 4 anos consecutivos para doutorado e pós-doutorado. Criar o processo Pessoal: Curso promovido por outra instituição. Nível de Acesso Restrito com Hipótese Legal Informação Pessoal.',
      'Incluir o documento Requerimento Geral iniciando com: "Venho requerer capacitação em serviço, para participação em programa de pós-graduação stricto sensu", especificar mestrado, doutorado ou pós-doutorado, nos termos da Portaria nº 32/2026. Incluir a proposta de redução da carga horária de 30%, período de realização, nome do curso e da instituição, e justificar a pertinência do tema e a impossibilidade de compensação de horas.',
      'Inserir declarações: declaração da instituição de ensino (admissão, nome do curso, plano de estudo, duração, título); declaração funcional do setorial de gestão de pessoas; e declaração de que não responde a processo administrativo disciplinar. Incluir o Termo de Compromisso usando o Documento Modelo número 193776278 (nível Público).',
      'Autorização das chefias: chefia imediata insere despacho fundamentado e assina; chefia mediata e Subcontrolador/Ouvidor-Geral/Controlador-Geral Adjunto assinam ou inserem novo despacho. Encaminhar o processo para a Unidade CGDF/SUBGI/COGEP/DIEST.',
    ],
    passosSEI: [
      {
        titulo: '1. Abrir o processo (Restrito)',
        itens: [
          'Tipo: <code>Pessoal: Curso promovido por outra instituição</code>.',
          'Nível de Acesso: <strong>Restrito</strong> — Hipótese Legal <em>Informação Pessoal</em>.',
        ],
      },
      {
        titulo: '2. Requerimento Geral',
        itens: [
          'Requerer a capacitação (mestrado/doutorado/pós-doutorado) nos termos da Portaria 32/2026.',
          'Propor a <strong>redução de até 30%</strong> da carga horária, período, curso, instituição e justificativa.',
        ],
      },
      {
        titulo: '3. Declarações + Termo de Compromisso',
        itens: [
          'Declaração da IES + declaração funcional da GP + declaração de ausência de PAD.',
          'Termo de Compromisso pelo <strong>Documento Modelo nº 193776278</strong>.',
        ],
      },
      {
        titulo: '4. Autorizações e envio',
        itens: [
          'Despachos das chefias imediata, mediata e da autoridade superior.',
          'Encaminhar para <code>CGDF/SUBGI/COGEP/DIEST</code>.',
        ],
      },
    ],
    dica: 'Requisitos de tempo: 3 anos consecutivos na CGDF para mestrado; 4 anos para doutorado e pós-doutorado.',
  },
];

// ── js/data/sigrh-demo.js ──
/**
 * sigrh-demo.js — Respostas simuladas da Fase 3 (integração SIGRH + SSO).
 *
 * IMPORTANTE: todos os valores aqui são FICTÍCIOS, apenas para demonstrar a
 * experiência de consulta autenticada da Fase 3. Nenhum dado real de servidor é
 * acessado ou armazenado. Na implementação real, estes dados viriam do SIGRH
 * somente após autenticação individual por SSO (Gov.br / GDF SSO), sob a
 * premissa de Security & Privacy by Design (LGPD).
 */

/** Consultas cujo texto contém estes termos são tratadas como "Fase 3". */
const GATILHOS_FASE3 = ['[fase 3]', 'sigrh', 'meu saldo', 'saldo individual'];

/**
 * @typedef {Object} RespostaFase3
 * @property {string} tipo
 * @property {string} titulo
 * @property {string} icone
 * @property {string} corpoHtml
 */

/** @type {Record<string, RespostaFase3>} */
const RESPOSTAS_FASE3 = {
  saldo: {
    tipo: 'saldo',
    titulo: 'Saldo funcional (SIGRH)',
    icone: 'fa-user-check',
    corpoHtml: `
      <p><strong>Servidor(a):</strong> [usuário autenticado via SSO]</p>
      <ul class="sei-steps">
        <li>🏖️ <strong>Saldo de férias:</strong> 15 dias restantes (período 2025/2026)</li>
        <li>📅 <strong>Abonos do exercício 2026:</strong> 3 dias disponíveis (2 usufruídos)</li>
        <li>🎂 <strong>Abono de aniversário (Lei 7.826/25):</strong> 1 dia pendente</li>
      </ul>
      <p>💡 Deseja ajuda para preencher o requerimento no SEI e agendar esses dias?</p>
    `,
  },
  aposentadoria: {
    tipo: 'aposentadoria',
    titulo: 'Simulação de aposentadoria (SIGRH)',
    icone: 'fa-calculator',
    corpoHtml: `
      <ul class="sei-steps">
        <li>⏱️ <strong>Tempo de serviço averbado:</strong> 24 anos, 6 meses e 12 dias</li>
        <li>🎯 <strong>Previsão de aposentadoria voluntária:</strong> 18/11/2031</li>
        <li>💰 <strong>Abono de permanência:</strong> elegibilidade prevista para 10/2029</li>
      </ul>
      <p>⚠️ Simulação informativa. Para contagem de tempo oficial averbada, abra processo no SEI para a GEREF.</p>
    `,
  },
};

/**
 * Decide se um texto de consulta deve ser tratado como Fase 3 e qual resposta usar.
 * @param {string} texto
 * @returns {RespostaFase3 | null}
 */
function resolverFase3(texto) {
  const t = texto.toLowerCase();
  const ehFase3 = GATILHOS_FASE3.some((g) => t.includes(g));
  if (!ehFase3) return null;
  if (t.includes('aposent') || t.includes('averb') || t.includes('tempo')) {
    return RESPOSTAS_FASE3.aposentadoria;
  }
  return RESPOSTAS_FASE3.saldo;
}

// ── js/data/impact.js ──
/**
 * impact.js — Métricas de impacto e evidências do projeto (aba Arquitetura).
 *
 * ATENÇÃO / INTEGRIDADE: por se tratar de uma proposta na categoria "Ideia"
 * (ainda não implementada), estes números são ESTIMATIVAS/PROJEÇÕES fundamentadas,
 * não resultados medidos. Os textos na UI deixam isso explícito para a Comissão
 * Julgadora, em linha com o edital (evidências e fundamentação).
 */

/**
 * @typedef {Object} Metrica
 * @property {string} valor
 * @property {string} rotulo
 * @property {string} detalhe
 * @property {string} icone
 * @property {'projecao'|'fato'} natureza
 */

/** @type {Metrica[]} */
const METRICAS_IMPACTO = [
  {
    valor: 'até 75%',
    rotulo: 'Redução de chamados repetitivos',
    detalhe: 'Projeção para dúvidas operacionais básicas hoje direcionadas à GEREF, GERFI e DIEST.',
    icone: 'fa-headset',
    natureza: 'projecao',
  },
  {
    valor: '24/7',
    rotulo: 'Atendimento disponível',
    detalhe: 'Orientação instantânea, inclusive fora do horário de expediente.',
    icone: 'fa-clock',
    natureza: 'fato',
  },
  {
    valor: '100%',
    rotulo: 'Abrangência na CGDF',
    detalhe: 'Servidores, estagiários e colaboradores — pontuação máxima de abrangência interna no edital.',
    icone: 'fa-users',
    natureza: 'fato',
  },
  {
    valor: '5 POPs',
    rotulo: 'Procedimentos já mapeados',
    detalhe: 'Férias, abono anual, abono aniversário, auxílio-alimentação e pós-graduação + LC 840/2011.',
    icone: 'fa-book',
    natureza: 'fato',
  },
];

/**
 * Evidências do problema (fundamentam a "Solução de problemas" do edital).
 * @type {Array<{titulo:string, texto:string, icone:string}>}
 */
const EVIDENCIAS_PROBLEMA = [
  {
    titulo: 'Normas dispersas',
    texto: 'Regras espalhadas entre a LC 840/2011, INs, portarias e POPs em versões diferentes dificultam a instrução correta no SEI.',
    icone: 'fa-diagram-predecessor',
  },
  {
    titulo: 'Retrabalho e devoluções',
    texto: 'Processos devolvidos por formulário errado, falta de anexo ou unidade de destino incorreta (ex.: GEREF x GERFI) geram atraso.',
    icone: 'fa-rotate-left',
  },
  {
    titulo: 'Sobrecarga da Gestão de Pessoas',
    texto: 'Equipes técnicas absorvem alto volume de perguntas repetitivas que poderiam ser autoatendidas com orientação padronizada.',
    icone: 'fa-people-arrows',
  },
];

/**
 * Dimensões ASG + ODS justificadas (o edital dá 8 pts quando há justificativa).
 * @type {Array<{dim:string, cor:string, icone:string, texto:string}>}
 */
const SUSTENTABILIDADE_ASG = [
  {
    dim: 'Governança (G)',
    cor: 'blue',
    icone: 'fa-scale-balanced',
    texto: 'Padroniza procedimentos e promove transparência ativa, reduzindo assimetria de informação entre servidor e Administração.',
  },
  {
    dim: 'Social (S)',
    cor: 'emerald',
    icone: 'fa-heart',
    texto: 'Dá autonomia ao servidor no acesso a direitos funcionais e reduz o estresse burocrático, valorizando o capital humano.',
  },
  {
    dim: 'Ambiental (A)',
    cor: 'teal',
    icone: 'fa-leaf',
    texto: 'Fortalece a cultura de processo 100% digital no SEI, eliminando impressões e deslocamentos desnecessários.',
  },
];

/** @type {Array<{num:string, titulo:string, meta:string}>} */
const ODS_ALINHADOS = [
  {
    num: 'ODS 16',
    titulo: 'Paz, Justiça e Instituições Eficazes',
    meta: 'Meta 16.6 — instituições eficazes, responsáveis e transparentes em todos os níveis.',
  },
  {
    num: 'ODS 9',
    titulo: 'Indústria, Inovação e Infraestrutura',
    meta: 'Meta 9.5 — incentivar a inovação e o desenvolvimento tecnológico.',
  },
];

// ── js/data/proposta.js ──
/**
 * proposta.js — Texto do Anexo I (Edital nº 04/2026) para copiar ao SEI.
 *
 * Mantido em um único lugar para servir tanto ao botão "Copiar Texto para o SEI"
 * quanto de referência ao conteúdo exibido na aba do formulário. As afirmações
 * de impacto são apresentadas como estimativas/projeções (categoria "Ideia").
 */

const PROPOSTA_ANEXO_I = `ANEXO I - PROPOSTA DE IDEIA OU INOVAÇÃO
3º Prêmio Inspiração de Inovação da CGDF (Edital nº 04/2026)

1. IDENTIFICAÇÃO DA PROPOSTA
a) Título: CGDF Digital: Assistente Virtual do Servidor (AViS)
b) Categoria: [ X ] Ideia
c) Segmento Temático: [ X ] Descomplica seu dia a dia
d) Proponente: [Nome Completo do Servidor] - Matrícula: [Número da Matrícula]
e) Unidade Administrativa: Controladoria-Geral do Distrito Federal - CGDF

2. DESCRIÇÃO DA PROPOSTA
f) Problema Identificado e Solução Proposta (em 3 Fases):
O Problema: servidores gastam tempo procurando regras na LC 840/2011 e nos POPs para instruir processos no SEI (férias v07, abono de 5 dias v3.0, abono de aniversário Lei 7.826/25, auxílio-alimentação v3.0 e pós-graduação Portaria 32/2026). Isso gera dúvidas repetitivas, processos devolvidos por instrução incorreta e sobrecarga nas equipes de Gestão de Pessoas (GEREF, GERFI, DIEST).
A Solução (Assistente Virtual com IA/RAG), em 3 Fases:
Fase 1 - MVP no Telegram conectado a um banco vetorial (Supabase/pgvector) com a carga da LC 840 e dos POPs, para orientação imediata e sem risco à privacidade (apenas dados públicos).
Fase 2 - Aplicação Web dedicada na Intranet da CGDF, com busca unificada e atalhos para os POPs mais consultados.
Fase 3 - Integração com o SIGRH mediante autenticação individual por SSO (Gov.br/GDF SSO) para dados funcionais sensíveis (saldo de férias, abonos, aposentadoria), sob Security & Privacy by Design (LGPD).

3. IMPACTO E BENEFÍCIOS ESPERADOS
g) Benefícios esperados (estimativas para a categoria Ideia): redução estimada de até 75% no volume de chamados repetitivos na Gestão de Pessoas; atendimento 24/7; menos devoluções no SEI graças à indicação exata de tipos de processo, formulários, prazos e unidade de destino; sigilo e rastreabilidade na consulta de dados pessoais.

4. PÚBLICO-ALVO
h) Abrangência: 100% dos servidores, estagiários e colaboradores da CGDF (abrangência institucional total), com potencial de replicação para os demais órgãos do GDF.

5. VIABILIDADE TÉCNICA E OPERACIONAL
i) Recursos e Replicabilidade: pilha flexível e de baixo custo (Telegram API + Supabase Vector DB na Fase 1; frontend web na Intranet na Fase 2; API Gateway + OAuth2/SSO na Fase 3). Existe protótipo funcional com recuperação (RAG) sobre os POPs reais, atestando a exequibilidade. Possível transição para execução 100% On-Premise com LLMs Open-Source locais (ex.: Llama 3/Qwen), garantindo soberania de dados.

6. SUSTENTABILIDADE (ASG / ODS)
j) Dimensões ASG e Agenda 2030: Governança (padronização e transparência ativa); Social (autonomia e valorização do servidor); Ambiental (processo 100% digital no SEI). Alinhamento justificado aos ODS 16 (Meta 16.6 - instituições eficazes e transparentes) e ODS 9 (Meta 9.5 - inovação e desenvolvimento tecnológico).`;

// ── js/rag/formatter.js ──
/**
 * formatter.js — Renderização das respostas do chatbot em HTML.
 *
 * Recebe o resultado do RagEngine e monta uma resposta clara e fundamentada:
 * passo a passo no SEI, prazos, base legal e — o diferencial de transparência —
 * um bloco "Fontes consultadas" com o PDF de origem, o score de relevância e o
 * trecho realmente recuperado. Também trata o caso de baixa confiança (fallback
 * honesto) e as respostas simuladas da Fase 3.
 */

/** Limite de confiança abaixo do qual acionamos o fallback honesto. */
const LIMIAR_CONFIANCA = 0.28;

/**
 * Escapa HTML para exibir texto do usuário com segurança.
 * @param {string} texto
 * @returns {string}
 */
function escaparHtml(texto) {
  return String(texto)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/**
 * Encurta um trecho para exibição na citação.
 * @param {string} texto
 * @param {number} [max]
 * @returns {string}
 */
function resumirTrecho(texto, max = 180) {
  const limpo = texto.trim();
  return limpo.length > max ? escaparHtml(limpo.slice(0, max)) + '…' : escaparHtml(limpo);
}

/**
 * Bloco "Fontes consultadas" — transparência da recuperação (evidência).
 * @param {import('./engine.js').RagEngine} */
function renderFontes(fontes) {
  if (!fontes || fontes.length === 0) return '';
  const itens = fontes
    .map((f) => {
      const pct = Math.round(Math.min(1, f.score / (f.score + 3)) * 100);
      return `
        <li class="fonte-item">
          <div class="fonte-cab">
            <i class="fa-solid fa-file-pdf"></i>
            <span class="fonte-nome">${escaparHtml(f.arquivoFonte)}</span>
            <span class="fonte-score" title="Relevância BM25">${pct}%</span>
          </div>
          <p class="fonte-trecho">"${resumirTrecho(f.trecho)}"</p>
        </li>`;
    })
    .join('');
  return `
    <details class="fontes-box">
      <summary><i class="fa-solid fa-magnifying-glass-chart"></i> Fontes consultadas (recuperação RAG)</summary>
      <ul class="fontes-lista">${itens}</ul>
    </details>`;
}

/**
 * Monta o HTML de uma resposta baseada em um documento da base.
 * @param {import('../data/knowledge-base.js').Documento} doc
 * @param {{confianca:number, fontes:any[]}} resultado
 * @returns {string}
 */
function renderResposta(doc, resultado) {
  const passos = doc.passosSEI
    .map(
      (p) => `
      <div class="passo-grupo">
        <p class="passo-titulo">${escaparHtml(p.titulo)}</p>
        <ul class="sei-steps">${p.itens.map((i) => `<li>${i}</li>`).join('')}</ul>
      </div>`,
    )
    .join('');

  const prazos = doc.prazos
    ? `<p class="bloco-prazos"><i class="fa-solid fa-hourglass-half"></i> <strong>Prazos:</strong> ${escaparHtml(doc.prazos)}</p>`
    : '';

  const dica = doc.dica ? `<p class="bloco-dica"><i class="fa-solid fa-lightbulb"></i> ${doc.dica}</p>` : '';

  return `
    <p class="resposta-titulo"><i class="fa-solid ${doc.icone}"></i> <strong>${escaparHtml(doc.titulo)}</strong>
      <span class="tag-versao">${escaparHtml(doc.versao)}</span></p>
    ${prazos}
    ${passos}
    <p class="bloco-destino"><i class="fa-solid fa-arrow-right-to-bracket"></i> <strong>Unidade de destino:</strong> <code>${escaparHtml(doc.unidadeDestino)}</code></p>
    <p class="bloco-legal"><span class="badge-art">Base legal</span> ${escaparHtml(doc.baseLegal)}</p>
    ${dica}
    ${renderFontes(resultado.fontes)}
  `;
}

/**
 * Resposta de fallback quando nenhuma passagem tem relevância suficiente.
 * @param {string} textoUsuario
 * @returns {string}
 */
function renderFallback(textoUsuario) {
  return `
    <p>Hmm, não achei nada sobre "<em>${escaparHtml(textoUsuario)}</em>" na minha base de POPs. 🤔</p>
    <p>Por enquanto cubro esses temas — toca num dos botões e eu te explico tudinho:</p>
    <div class="inline-actions-group">
      <button class="inline-btn" data-prompt="Como marcar, remarcar ou suspender minhas férias no SEI?">🏖️ Férias</button>
      <button class="inline-btn" data-prompt="Como peço o abono de ponto anual de 5 dias?">📅 Abono 5 dias</button>
      <button class="inline-btn" data-prompt="Como funciona o abono no dia do meu aniversário (Lei 7.826/2025)?">🎂 Abono aniversário</button>
      <button class="inline-btn" data-prompt="Como solicito o auxílio alimentação/refeição pelo SEI?">🍽️ Auxílio alimentação</button>
      <button class="inline-btn" data-prompt="Quais as regras para pós-graduação stricto sensu (Portaria 32/2026)?">🎓 Pós-graduação</button>
    </div>
  `;
}

/**
 * Resposta simulada da Fase 3 (dados fictícios), sempre com aviso claro.
 * @param {import('../data/sigrh-demo.js').RespostaFase3} resposta
 * @returns {string}
 */
function renderFase3(resposta) {
  return `
    <div class="fase3-banner">
      <p class="fase3-tag"><i class="fa-solid fa-lock"></i> Fase 3 — Consulta autenticada por SSO (Gov.br / GDF SSO)</p>
      <p class="fase3-aviso"><i class="fa-solid fa-circle-info"></i> Demonstração com <strong>dados fictícios</strong>. Na versão real, o acesso ocorre só após login individual, sob Security &amp; Privacy by Design (LGPD).</p>
    </div>
    <p class="resposta-titulo"><i class="fa-solid ${resposta.icone}"></i> <strong>${escaparHtml(resposta.titulo)}</strong></p>
    <div class="fase3-corpo">${resposta.corpoHtml}</div>
  `;
}

// ── js/chat.js ──
/**
 * chat.js — Controlador da conversa.
 *
 * Liga a interface (dois canais: Intranet e Telegram) ao motor RAG. Cuida de
 * renderizar mensagens, exibir o indicador de digitação, espelhar a conversa
 * entre os canais e decidir a resposta (Fase 3 simulada, recuperação RAG ou
 * fallback honesto).
 */

function horaFormatada() {
  const agora = new Date();
  const h = String(agora.getHours()).padStart(2, '0');
  const m = String(agora.getMinutes()).padStart(2, '0');
  return `${h}:${m}`;
}

class ChatController {
  /**
   * @param {{
   *   intraBody: HTMLElement, intraInput: HTMLInputElement,
   *   tgBody: HTMLElement, tgInput: HTMLInputElement,
   *   onEnviar?: () => void,
   * }} refs
   */
  constructor(refs) {
    this.refs = refs;
    this.engine = new RagEngine(BASE_CONHECIMENTO);
  }

  /**
   * Decide a resposta HTML para um texto do usuário.
   * @param {string} texto
   * @returns {string}
   */
  gerarResposta(texto) {
    const fase3 = resolverFase3(texto);
    if (fase3) return renderFase3(fase3);

    const resultado = this.engine.query(texto);
    if (!resultado.documento || resultado.confianca < LIMIAR_CONFIANCA) {
      return renderFallback(texto);
    }
    return renderResposta(resultado.documento, resultado);
  }

  /**
   * Renderiza uma mensagem num container.
   * @param {HTMLElement} container
   * @param {string} html
   * @param {boolean} ehUsuario
   */
  _append(container, html, ehUsuario) {
    const div = document.createElement('div');
    div.className = `message ${ehUsuario ? 'user-msg' : 'bot-msg'}`;

    if (!ehUsuario && container === this.refs.intraBody) {
      div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-content">
          <div class="msg-header-name">AViS CGDF</div>
          ${html}
          <span class="msg-time">${horaFormatada()}</span>
        </div>`;
    } else if (!ehUsuario) {
      div.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-content">${html}<span class="msg-time">${horaFormatada()}</span></div>`;
    } else {
      div.innerHTML = `
        <div class="msg-content">${html}<span class="msg-time">${horaFormatada()}</span></div>`;
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  /**
   * Indicador "digitando…".
   * @param {HTMLElement} container
   * @returns {HTMLElement}
   */
  _typing(container) {
    const div = document.createElement('div');
    div.className = 'message bot-msg typing-msg-temp';
    div.innerHTML = `
      <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
      <div class="msg-content">
        <div class="typing-indicator">
          <span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>
        </div>
      </div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
  }

  /**
   * Fluxo completo de envio: eco do usuário, digitação, resposta e espelhamento.
   * @param {HTMLInputElement} inputEl
   * @param {HTMLElement} bodyEl
   */
  enviar(inputEl, bodyEl) {
    const texto = inputEl.value.trim();
    if (!texto) return;

    this._append(bodyEl, `<p>${escaparHtml(texto)}</p>`, true);
    inputEl.value = '';

    const typing = this._typing(bodyEl);

    // Pequeno atraso simula o tempo de recuperação + geração (UX de chat real).
    setTimeout(() => {
      typing.remove();
      const respostaHtml = this.gerarResposta(texto);
      this._append(bodyEl, respostaHtml, false);

      // Espelha para o outro canal, mantendo as duas telas coerentes na demo.
      const outro = bodyEl === this.refs.tgBody ? this.refs.intraBody : this.refs.tgBody;
      if (outro) {
        this._append(outro, `<p>${escaparHtml(texto)}</p>`, true);
        this._append(outro, respostaHtml, false);
      }
    }, 500);
  }

  /**
   * Dispara uma pergunta pré-definida no canal atualmente ativo.
   * @param {string} prompt
   * @param {'intranet'|'telegram'} canal
   */
  enviarPrompt(prompt, canal) {
    if (canal === 'telegram' && this.refs.tgInput && this.refs.tgBody) {
      this.refs.tgInput.value = prompt;
      this.enviar(this.refs.tgInput, this.refs.tgBody);
    } else if (this.refs.intraInput && this.refs.intraBody) {
      this.refs.intraInput.value = prompt;
      this.enviar(this.refs.intraInput, this.refs.intraBody);
    }
  }
}

// ── js/ui.js ──
/**
 * ui.js — Comportamentos de interface independentes do chat.
 *
 * Agrupa: notificações (toasts), navegação por abas, alternância de tema
 * claro/escuro (com persistência), troca de canal (Intranet/Telegram), filtro
 * dos cards de POP e o atalho de teclado Ctrl+K. Cada função é isolada e
 * defensiva (verifica a existência dos elementos antes de agir).
 */

const CHAVE_TEMA = 'avis-tema';

/** Cria o sistema de toasts e devolve a função de exibição. */
function criarToasts() {
  const container = document.getElementById('toastContainer');
  return function showToast(mensagem, icone = 'fa-circle-check', tipo = 'info') {
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${tipo}`;
    toast.setAttribute('role', 'status');
    toast.innerHTML = `<i class="fa-solid ${icone}"></i> <span>${mensagem}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(8px)';
      toast.style.transition = 'all 0.25s ease';
      setTimeout(() => toast.remove(), 250);
    }, 3000);
  };
}

/** Navegação por abas com atualização de ARIA. */
function initTabs() {
  const botoes = document.querySelectorAll('.nav-btn');
  const conteudos = document.querySelectorAll('.tab-content');
  botoes.forEach((btn) => {
    btn.addEventListener('click', () => {
      const alvo = btn.getAttribute('data-tab');
      botoes.forEach((b) => {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      conteudos.forEach((c) => c.classList.remove('active'));
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const secao = document.getElementById(`tab-${alvo}`);
      if (secao) secao.classList.add('active');
    });
  });
}

/**
 * Alternância de tema com persistência em localStorage.
 * @param {(msg:string, icone?:string)=>void} showToast
 */
function initTema(showToast) {
  const botao = document.getElementById('themeToggle');
  if (!botao) return;

  const aplicar = (escuro, silencioso = false) => {
    if (escuro) {
      document.body.setAttribute('data-theme', 'dark');
      botao.innerHTML = '<i class="fa-solid fa-sun"></i>';
      if (!silencioso) showToast('Modo Escuro ativado', 'fa-moon');
    } else {
      document.body.removeAttribute('data-theme');
      botao.innerHTML = '<i class="fa-solid fa-moon"></i>';
      if (!silencioso) showToast('Modo Claro ativado', 'fa-sun');
    }
    try {
      localStorage.setItem(CHAVE_TEMA, escuro ? 'dark' : 'light');
    } catch (_) {
      /* localStorage indisponível — segue sem persistir */
    }
  };

  let inicial = true; // escuro por padrão
  try {
    const salvo = localStorage.getItem(CHAVE_TEMA);
    if (salvo !== null) inicial = salvo === 'dark';
  } catch (_) {
    /* localStorage indisponível — mantém padrão escuro */
  }
  aplicar(inicial, true);

  botao.addEventListener('click', () => {
    const escuroAgora = document.body.getAttribute('data-theme') === 'dark';
    aplicar(!escuroAgora);
  });
}

/**
 * Troca de canal (Intranet x Telegram).
 * @param {(msg:string, icone?:string)=>void} showToast
 * @returns {() => 'intranet'|'telegram'} função que devolve o canal ativo
 */
function initCanais(showToast) {
  const botoes = document.querySelectorAll('.segment-btn');
  const telegramFrame = document.getElementById('telegramFrame');
  const intranetFrame = document.getElementById('intranetFrame');

  botoes.forEach((btn) => {
    btn.addEventListener('click', () => {
      const canal = btn.getAttribute('data-channel');
      botoes.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      if (canal === 'telegram') {
        telegramFrame?.classList.add('active-frame');
        intranetFrame?.classList.remove('active-frame');
        showToast('Alternado para o Telegram Bot', 'fa-brands fa-telegram');
      } else {
        intranetFrame?.classList.add('active-frame');
        telegramFrame?.classList.remove('active-frame');
        showToast('Alternado para a Aplicação Web Intranet', 'fa-globe');
      }
    });
  });

  return function canalAtivo() {
    const ativo = document.querySelector('.segment-btn.active');
    return ativo && ativo.getAttribute('data-channel') === 'telegram' ? 'telegram' : 'intranet';
  };
}

/**
 * Filtro ao vivo dos cards de POP + atalho Ctrl+K.
 * @param {(msg:string, icone?:string)=>void} showToast
 */
function initBuscaPops(showToast) {
  const input = document.getElementById('popSearchInput');
  const cards = document.querySelectorAll('.pop-card');
  if (!input) return;

  input.addEventListener('input', (e) => {
    const q = e.target.value.toLowerCase().trim();
    cards.forEach((card) => {
      card.style.display = card.textContent.toLowerCase().includes(q) ? 'flex' : 'none';
    });
  });

  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      input.focus();
      showToast('Pesquisa focada (Ctrl + K)', 'fa-magnifying-glass');
    }
  });
}

// ── js/app.js ──
/**
 * app.js — Ponto de entrada do protótipo AViS CGDF (ES module).
 *
 * Orquestra a inicialização: interface (abas, tema, canais, busca), o
 * controlador de chat com o motor RAG, e a renderização dos blocos de impacto,
 * evidências e sustentabilidade (dados em js/data/impact.js).
 *
 * DEV LOCAL: por usar ES modules, abra via um servidor HTTP — não por file://.
 *   Ex.:  python -m http.server 8000   →   http://localhost:8000
 * Em produção (GitHub Pages) funciona diretamente, pois é servido por HTTPS.
 */

/** Preenche um container (se existir) com HTML. */
function preencher(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

/** Renderiza métricas, evidências e ASG a partir dos dados. */
function renderConteudoEstatico() {
  preencher(
    'metricsGrid',
    METRICAS_IMPACTO.map(
      (m) => `
      <div class="metric-card">
        <div class="metric-icon"><i class="fa-solid ${m.icone}"></i></div>
        <div class="metric-num">${m.valor}</div>
        <div class="metric-label">${m.rotulo}</div>
        <p class="metric-detail">${m.detalhe}</p>
        ${m.natureza === 'projecao' ? '<span class="metric-tag">estimativa</span>' : ''}
      </div>`,
    ).join(''),
  );

  preencher(
    'evidenceList',
    EVIDENCIAS_PROBLEMA.map(
      (e) => `
      <div class="evidence-item">
        <i class="fa-solid ${e.icone}"></i>
        <div><h4>${e.titulo}</h4><p>${e.texto}</p></div>
      </div>`,
    ).join(''),
  );

  preencher(
    'asgGrid',
    SUSTENTABILIDADE_ASG.map(
      (a) => `
      <div class="asg-card asg-${a.cor}">
        <i class="fa-solid ${a.icone}"></i>
        <h4>${a.dim}</h4>
        <p>${a.texto}</p>
      </div>`,
    ).join(''),
  );

  preencher(
    'odsList',
    ODS_ALINHADOS.map(
      (o) => `
      <div class="ods-item">
        <span class="ods-num">${o.num}</span>
        <div><strong>${o.titulo}</strong><p>${o.meta}</p></div>
      </div>`,
    ).join(''),
  );
}

/** Liga os botões de envio, Enter, atalhos e delegação de cliques. */
function initChat(showToast) {
  const refs = {
    intraBody: document.getElementById('intraChatBody'),
    intraInput: document.getElementById('intraInput'),
    tgBody: document.getElementById('tgChatBody'),
    tgInput: document.getElementById('tgInput'),
  };
  const chat = new ChatController(refs);
  const canalAtivo = initCanais(showToast);

  const intraSend = document.getElementById('intraSendBtn');
  const tgSend = document.getElementById('tgSendBtn');

  if (intraSend && refs.intraInput && refs.intraBody) {
    intraSend.addEventListener('click', () => chat.enviar(refs.intraInput, refs.intraBody));
    refs.intraInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') chat.enviar(refs.intraInput, refs.intraBody);
    });
  }
  if (tgSend && refs.tgInput && refs.tgBody) {
    tgSend.addEventListener('click', () => chat.enviar(refs.tgInput, refs.tgBody));
    refs.tgInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') chat.enviar(refs.tgInput, refs.tgBody);
    });
  }

  // Limpar conversa da intranet.
  const limpar = document.getElementById('clearChatBtn');
  if (limpar && refs.intraBody) {
    limpar.addEventListener('click', () => {
      refs.intraBody.innerHTML = `
        <div class="message bot-msg">
          <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
          <div class="msg-content">
            <div class="msg-header-name">AViS</div>
            <p>Pronto, limpei a conversa! 😊 Me conta sua dúvida sobre férias, abonos ou auxílios que eu te ajudo.</p>
          </div>
        </div>`;
      showToast('Histórico de chat limpo', 'fa-rotate-left');
    });
  }

  // Delegação: chips da sidebar, cards de POP, teclado do Telegram e botões inline.
  document.addEventListener('click', (e) => {
    const gatilho = e.target.closest('.prompt-chip, .pop-card, .tg-kb-btn, .inline-btn');
    if (!gatilho) return;
    const prompt = gatilho.getAttribute('data-prompt');
    if (prompt) chat.enviarPrompt(prompt, canalAtivo());
  });
}

/** Botão "Copiar Texto para o SEI" (aba Anexo I). */
function initCopiarProposta(showToast) {
  const botao = document.getElementById('copyProposalBtn');
  if (!botao) return;
  botao.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(PROPOSTA_ANEXO_I);
      showToast('Anexo I copiado para a área de transferência!', 'fa-copy', 'success');
    } catch (_) {
      showToast('Não foi possível copiar automaticamente.', 'fa-triangle-exclamation', 'error');
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const showToast = criarToasts();
  initTabs();
  initTema(showToast);
  initBuscaPops(showToast);
  renderConteudoEstatico();
  initChat(showToast);
  initCopiarProposta(showToast);
});

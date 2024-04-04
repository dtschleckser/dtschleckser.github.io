import AltText from '../components/AltText'
import ExpandableText from '../components/ExpandableText'

import moeArch from '../assets/moe-arch.png'
import denseVsMoe from '../assets/dense-vs-moe.png'
import mixtralLayerExperts from '../assets/mixtral-layer-experts.png'
import routerScalingLoss from '../assets/router-scaling-loss.png'

import moeHeaderArt from '../assets/blog-headers/moe-gradient.png'

import { InlineMath, BlockMath } from 'react-katex';

import { Code, Title, Divider, Text } from '@mantine/core';

import { FaArrowLeft } from "react-icons/fa6";
import { Link } from 'react-router-dom'

function MixtureOfExperts() {
    
  const cvDemo = `CV of [0.09, 0.8, 0.01, 0.1] = 1.63
  CV of [0.25, 0.25, 0.25, 0.25] = 0.0
  CV of [0.2, 0.3, 0.2, 0.3] = 0.04`
  
    return (
      <div style={{ textAlign: 'left', width: '100%' }}>

        <Link to='/' style={{ display: 'flex', marginBottom: '1rem', paddingTop: '2rem', alignItems: 'center' }}>
          <FaArrowLeft style={{marginRight: '5px', marginTop: '2px'}}/>
          <h3>Back</h3>
        </Link>

        <Title order={1}>
          An Introduction to Sparsely Gated MoE
        </Title>
        <Text c="dimmed" size="lg" style={{ paddingBottom: '1rem' }}>
          April 4, 2024
        </Text>
        <center>
          <img src={moeHeaderArt} style={{ width: '70%', marginTop: '1rem',  marginBottom: '1rem' }} className='dropShadowImage' />
        </center>
        <div>
          MoE (mixture of experts) models have experienced a significant increase in attention and popularity
          in recent months. Both open and closed-source LLMs have been trending towards sparsely
          gated mixture of expert models, with
          <AltText text="GPT-4" alt="There's very little official information about GPT-4's architecture.
          It's not confirmed to be a MoE, but NVIDIA's 2024 keynote featured a 'GPT-MoE-1.8T' model, 
          likely GPT-4." link="https://youtu.be/Y2F8yisiS6E?t=1193" /> and
          {' '}<a href="https://huggingface.co/mistralai/Mixtral-8x7B-v0.1">Mixtral-8x7b</a>{' '}
          enjoying both high performance and fast inference relative to their total parameter count.
          So what does a "mixture of experts" do? How does it work and why would we want to use it?
        </div>
  
        <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
          Sparsity
        </Title>
        <div>
          Parameter count is valuable in LLMs.
          {' '}<a href="https://arxiv.org/pdf/2001.08361.pdf">Neural scaling laws</a>{' '}
          show us that we can significantly improve the performance of an LLM by scaling out, adding more and larger layers.
          However, it comes at a clear cost. The more parameters you have, the compute you need for both training and inference.
          What if we could keep a significantly greater number of parameters and only use the ones that
          are most relevant to our particular input? This is the observation and motivation behind sparsity in networks.
        </div>
        <br />
        <div>
          Sparsity allows us to selectively activate and compute only a subset of the model's parameters for each input,
          reducing the computational cost and memory footprint during inference. By leveraging sparsity, we can build
          larger models with more parameters while keeping the actual computation and memory usage manageable.
        </div>
        <br />
        <div>
        <div>
          The basic concept behind MoE has been around for a while, but a prominent recent example is
          {' '}<a href="https://arxiv.org/pdf/1701.06538.pdf">The Sparsely-Gated Mixture of Experts Layer</a>.
          In this paper, the authors introduce the concept of an MoE layer in the context of RNNs - let's
          see what the architecture looks like.
        </div>
        
        </div>

        <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
          The Mixture of Experts Layer
        </Title>
        <div>
          Instead of passing our input through a standard dense feedforward network, the MoE layer routes the input to a given number of
          "expert" networks. We express the list of experts we're using as{' '}
          <InlineMath>{'\\{E_0, E_1, \\ldots, E_{n-1}\\}'}</InlineMath> when we have <InlineMath>n</InlineMath> total experts.


          <br /><br />

          Each expert (in the above paper) is a 2-layer FFN with input of size 1024, output of size 512,
          and a ReLU activation function. Each of these experts gives their own unique output, and we can choose how many experts we want to use
          in each forward pass by sending our input out to each of them, and then summing their output afterwards. The quantity of experts we use
          is a hyperparameter usually called <Code>top_k_experts</Code>. 
          <br /><br />

          Since we only use a subset of our experts for a single item / forward pass, not all of our parameters are going to be "active". As such,
          we call the parameters that are actively being used "active parameters", and the total parameters in the model,
          including those that aren't used for a given forward pass, the "total" or "sparse" parameter count. Depending on our expert count
          and <Code>top_k_expert</Code> configuration, these can be significantly different numbers. In a traditional dense network,
          they'd be the same, as all the parameters are used for every forward pass. 

          <br /><br />

          To choose the experts we think will be best, we use a gating network - we'll call it <InlineMath>G(x)</InlineMath>.
          There are various ways of defining how <InlineMath>G(x)</InlineMath> will actually work that we'll elaborate on later in this post,
          but for now we can assume it will black-box select the set of experts that are best to handle our input.
        </div>
        <br />
        To use our gating network to select experts, we'll use the formula:
        <div className='centeredMath'>
            <InlineMath>{'y = \\sum_{i=1}^{n} G(x)_i E_i(x)'}</InlineMath>
        </div>
        For each expert <InlineMath>E_i(x)</InlineMath> network, we determine whether
        to use them with the gating network <InlineMath>G(x)_i</InlineMath>'s value at index <InlineMath>i</InlineMath>.
        If <InlineMath>G(x)_i</InlineMath> is 0, we don't need to calculate the value of the expert <InlineMath>E_i(x)</InlineMath> at all!
        Assuming we get <InlineMath>G(x)</InlineMath> into a nice distribution, we can select the top <InlineMath>k</InlineMath> experts,
        ensuring that we have a value for <InlineMath>k</InlineMath> that is low enough to be sparse but high enough to add enough information
        from our experts.
        <br /><br />
        With all this information in mind, here's the original MoE layer diagram from Shazeer et al.:
  
        <img src={moeArch} style={{ width: 'inherit' }}/>
  
        We can see our input <InlineMath>x</InlineMath> is being passed through a gating network and two experts are selected;
        they're being multiplied by the gating values <InlineMath>{'G(x)_2'}</InlineMath> and <InlineMath>{'G(x)_\{n-1\}'}</InlineMath> 
        (intuitively, the gating values correspond to the importance of those experts to the output).
        This also makes the gating network differentiable. Then, they're all summed and that's the output of the MoE layer!
  
        <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
          Modifications for Transformers
        </Title>
        In a Transformer model, we're dealing with transformer blocks that consist of a self-attention block and a FFN.
        When applying MoE to Transformers, we replace the standard single FFN in a Transformer block with a mixture of experts mechanism that has multiple expert FFNs.
        Similarly to the RNN case, we select the most appropriate one with a gating mechanism.
        
        <img src={denseVsMoe} style={{ width: 'inherit', paddingTop: '1rem', paddingBottom: '1rem' }}/>
        
        This figure is taken from <a href="https://arxiv.org/pdf/2209.01667.pdf">A Review of Sparse Expert Models in Deep Learning</a>.
        
        <br />
        
        For a practical example of MoE transformer implementation, I'd recommend
        checking out the <a href="https://github.com/mistralai/mistral-src/blob/main/mistral/model.py#L145">Mixtral reference implementation</a>.
  
        <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
          How to Train Your Gating Network
        </Title>
  
        Choosing an appropriate architecture and loss to train our gating network is important, as we want
        to ensure that each expert ends up with a roughly even responsibility. We don't want a single expert that
        the gating network always routes to and receives all the data during training; this problem compounds during training,
        as the router will greatly prefer the expert that has seen more data.
        <br /><br />
        In the below cases, our gating network
        is trained via backprop like the rest of the network, but there are some cases where the gating network isn't trained
        that way (e.g. <a href="https://arxiv.org/pdf/2106.04426.pdf">hash layers</a>). 
  
        <Title order={3} style={{ paddingTop: '2rem' }}>
          Gating methods
        </Title>
        <ExpandableText headerText="Softmax routing">
          A basic routing network that doesn't address this problem may look like a learnable matrix <InlineMath>{'W_g'}</InlineMath> into softmax:
          <div className='centeredMath'>
            <InlineMath>{'G(x) = Softmax(x \\cdot W_g)'}</InlineMath>
          </div>
          We take the input and multiply it with our learned matrix <InlineMath>{'W_g'}</InlineMath>, and then softmax to ensure we're able to
          scale evenly.
        </ExpandableText>
  
        <ExpandableText headerText="Noisy Top-K Gating">
          To start addressing the problem of load balancing our experts, we can
          add noise to our pre-softmax gating logits to even out the probability distribution
          and keep the top K outputs in order to make it sparse. The noise will be leveraged to a greater
          extent in the load balancing loss we'll describe later.
          <br /><br />
          First, the noise is added; it's scaled with a learnable matrix <InlineMath>{'W_\{noise\}'}</InlineMath>{' '}
          so that the amount of noise we add is appropriate for our given input:
          <div className='centeredMath'>
            <InlineMath>{'Noise(x) = StandardNormal() \\cdot Softplus(x \\cdot W_\{noise\})'}</InlineMath>
          </div>
          StandardNormal is just the standard normal distribution and Softplus is used to keep the outputs positive.
          We add the noise to the outputs of our gating network
          <div className='centeredMath'>
            <InlineMath>{'H(x)_i = (x \\cdot W_g)_i + Noise(x)'}</InlineMath>
          </div>
          After this, we can "KeepTopK" (simply keeping the greatest K values of x unmodified, with the rest set to -infinity).
          Then, we softmax the output of KeepTopK to get our selection of top K experts:
          <div className='centeredMath'>
            <InlineMath>{'G(x) = Softmax(KeepTopK(H(x), k))'}</InlineMath>
          </div>
        </ExpandableText>
  
        <Title order={3} style={{ paddingTop: '1rem' }}>
          Losses
        </Title>
  
        <ExpandableText headerText="Importance Loss + Load Balancing Loss">
          Separately, Shazeer et al. add an auxiliary loss <InlineMath>{'L_{importance}'}</InlineMath> that's
          defined over a batch of data. They take the "importance value" of an expert, which is the sum
          of the values of the gating network (i.e. post-softmax) in that batch:
          <div className='centeredMath'>
            <BlockMath>{'Importance(X) = \\sum_{x \\in X} G(x)'}</BlockMath>
          </div>
          Then, they add the squared <AltText text="coefficient of variation" alt="CV(X) = stdev(x) / mean(x)"
          link="https://en.wikipedia.org/wiki/Coefficient_of_variation#Definition" /> of the data, scaled by a
          hyperparameter <InlineMath>{'w_{importance}'}</InlineMath>:
          <div className='centeredMath'>
            <InlineMath>{'L_{importance}(X) = w_{importance} \\cdot CV(Importance(X))^2'}</InlineMath>
          </div>
  
          The coefficient of variation punishes an uneven distribution and rewards an even one:
          <center>
            <Code block style={{ width: 'fit-content', textAlign: 'start', marginTop: '1rem', marginBottom: '1rem' }}>{cvDemo}</Code>
          </center>
          As it's an auxiliary loss, this is added to the standard loss for the network.
          <Divider size="md" style={{ marginTop: '1rem', marginBottom: '1rem'}}/>
          They also introduce a load balancing loss. It uses an estimator <InlineMath>{'Load(x)_i'}</InlineMath>, that
          is defined with data batch <InlineMath>{'X'}</InlineMath> for a single expert <InlineMath>{'i'}</InlineMath> and
          a probability <InlineMath>{'P(x, i)'}</InlineMath>:
          <div className='centeredMath'>
            <BlockMath>{'Load(X)_i = \\sum_{x \\in X} P(x, i)'}</BlockMath>
          </div>
          The exact definition for <InlineMath>{'P(x, i)'}</InlineMath> is fairly lengthy, but it's defined in terms that
          we've seen before. 
          <div className='centeredMath'>
            <InlineMath>{'P(x, i) = Pr((x \\cdot W_g)_i + StandardNormal() \\cdot Softplus((x \\cdot W_{noise})_i) > kth\\_excluding(H(x), k, i))'}</InlineMath>
          </div>
          You may recognize the term <InlineMath>{'(x \\cdot W_g)_i + StandardNormal() \\cdot Softplus((x \\cdot W_{noise})_i)'}</InlineMath> as{' '}
          <InlineMath>{'H(x)'}</InlineMath>, the noised outputs from our gating network, from the noisy top K gating function above. 
          <br /><br />
          The concept behind <InlineMath>{'P(x, i)'}</InlineMath> is to calculate the probability that
          our item <InlineMath>{'i'}</InlineMath> will pass the gating function given a specific expert. 
          We want it to be a close competition between experts, rather than routing all tokens to a single
          one, so we check the probability that the noised item going through
          expert <InlineMath>{'i'}</InlineMath> will make the cut for the top K experts
          (the probability comes from our scaled StandardNormal).
          <br /><br />
          Once we have our load function defined, we can get our final auxiliary loss by calculating 
          the squared CV, scaled by a hyperparameter <InlineMath>{'w_{load}'}</InlineMath>,
          very similarly to our importance loss:
          
          <div className='centeredMath'>
            <BlockMath>{'L_{load}(X) = w_{load} \\cdot  CV(Load(X))^2'}</BlockMath>
          </div>
          
          We want <InlineMath>{'Load(X)'}</InlineMath>, our measure of the likelihood that the values
          of <InlineMath>{'X'}</InlineMath> pass the gate, to be even across all our items in <InlineMath>{'X'}</InlineMath> and
          across all our experts.
        </ExpandableText>
  
        <ExpandableText headerText="Switch Transformer Load Balancing Loss">
          <a href="https://arxiv.org/pdf/2101.03961.pdf">Switch transformers</a>{' '}
          greatly simplifies the load balancing + importance loss introduced by Shazeer et al. It introduces 
          an auxiliary loss to ensure that the load is balanced across experts. For:
          <ul>
            <li>Tokens <InlineMath>{'T'}</InlineMath> in the batch</li>
            <li>Quantity of experts <InlineMath>{'N'}</InlineMath>,</li>
            <li>Scaling hyperparameter <InlineMath>{'\\alpha'}</InlineMath> (defined as <InlineMath>{'10^{-2}'}</InlineMath> in the paper),</li>
          </ul>
          the auxiliary loss is defined by 
          <BlockMath>{'loss = \\alpha \\cdot N \\cdot \\sum_{i=1}^{N} f_i \\cdot P_i '}</BlockMath>
          where: <br />
          <InlineMath>{'f_i'}</InlineMath> is the fraction of tokens that get routed to expert <InlineMath>{'i'}</InlineMath> in the batch <InlineMath>{'\\beta'}</InlineMath>, and <br />
          <InlineMath>{'P_i'}</InlineMath> is the fraction of all router probability dedicated to expert <InlineMath>{'i'}</InlineMath> for all tokens in the batch <InlineMath>{'\\beta'}</InlineMath>.
          <br /> <br />
          Essentially, we want the router to have the token quantity and the routing probability to be even across all our experts.
  
        </ExpandableText>

        <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '3rem' }}>
          Scaling Laws
        </Title>
        <div>
          We can get a reasonably good idea of how a transformer model will perform on a given set of data
          if we have a known parameter count and quantity of compute. How does it apply to MoE models? The
          paper "<a href="https://arxiv.org/pdf/2202.01169.pdf">Unified Scaling Laws for Routed Language Models"</a>
          seeks to define scaling laws that show how we can expect performance to improve when we scale our expert count
          (and therefore total parameter count) up.
          <br />
          { /* https://arxiv.org/pdf/2202.01169.pdf */ }
          <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem 0 2rem 0'}}>
            <img src={ routerScalingLoss } style={{ maxWidth: '500px' }} />
          </div>
          Each point on an individual curve on the above graph is a model configuration that is equally performant
          when compared to another point on the same curve (as measured by loss).
          We can see that, for example, a 32-expert 5M param model is roughly similar in performance to a dense (i.e. one-expert)
          25M model. We also see diminishing performance when scaling to large quantities of experts
          (even with the exponential expert count axis).
        </div>
  
        <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '2rem' }}>
          Open source MoE LLMs
        </Title>
  
        <div>
          A good example of a recent, high-performing, open-source MoE model is Mixtral. It has 46.7B total params, 12.9B of which
          are active (i.e. used during inference), and is highly competitive on benchmarks with non-MoE Transformer models like LLaMA 2 70B,
          while using 66% of the active parameter count. The primary drawback is the high VRAM cost, as all parameters need to be loaded
          in memory even if we're only using a subset - we're not likely to use all the same experts when generating even a couple tokens, so
          offloading experts wouldn't be efficient.
          <br />
          <br />
          The number of experts can vary greatly in open MoE LLMs. The
          largest Switch Transformer model <a href="https://huggingface.co/google/switch-c-2048">switch-c-2048</a> has 2048 experts,
          resulting in a model with over a trillion active parameters. In comparison, Mixtral
          has <Code>top_k_experts = 2</Code> and <Code>n_experts = 8</Code>. Recent releases are closer to the Mixtral expert count - {' '}
          <a href="https://github.com/xai-org/grok-1">Grok-1</a> has 8 experts with 2 active
          and <a href="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm">DBRX</a> has 16 with 4 active.  
        </div>

        <h2>What do the experts do?</h2>

        At least in the case of <a href="https://arxiv.org/pdf/2202.08906.pdf">ST-MoE</a> (section 7.3) and Mixtral,
        it doesn't look like experts correspond cleanly to a single human-interpretable topic.
        They seem to focus more on token-level features: The <a href="https://arxiv.org/pdf/2401.04088.pdf">Mixtral paper</a> shows
        that tokens such as commas and operators in Python are handled by one expert,
        while whitespace is handled by another (and this is roughly true throughout all layers).
  
        <br />
        <br />
  
        We can also check what experts Mixtral chooses for different datasets covering domains like math, code, and science.
        They find that each expert covers a roughly similar amount of tokens from the documents in that domain (also from the Mixtral paper):
  
        <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem 0 2rem 0'}}>
          <img src={ mixtralLayerExperts } style={{ width: '100%', maxWidth: '1000px', paddingTop: '1rem', paddingBottom: '1rem' }}/>
        </div>
        <Title order={2} style={{ paddingBottom: '1rem', paddingTop: '2rem' }}>
          Conclusion
        </Title>
  
        <div>
          Mixture of expert architectures are a powerful tool that have gained prominence, particularly in large language models.
          MoE architectures enable training larger models that are able to leverage sparsity in their computation. Despite larger memory
          requirements for the model's total parameters
          <br />
          <br />
          By leveraging sparsely activated experts, MoE models can achieve impressive performance while keeping computation costs
          manageable compared to dense models of similar total parameter counts. In the near future, I expect to see more large
          MoE releases and additional tricks to exploit sparsity in large models.
        </div>

        <Link to='/' style={{ display: 'flex', marginBottom: '1rem', paddingTop: '2rem', alignItems: 'center' }}>
          <FaArrowLeft style={{marginRight: '5px', marginTop: '2px'}}/>
          <h3>Back</h3>
        </Link>

        <br /><br />
      </div>
    )
}

export default MixtureOfExperts
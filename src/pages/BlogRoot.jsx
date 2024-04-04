import moeHeaderArt from '../assets/blog-headers/moe-gradient.png'
import BlogListItem from '../components/BlogListItem/BlogListItem'

function BlogRoot() {
    return <div>
        <h2>Posts</h2>
        <ul style={{ listStyleType: 'none', margin: '0px', padding: '0px' }}>
            <li>
                <BlogListItem
                    image={moeHeaderArt}
                    title="An Introduction to Sparsely Gated MoE"
                    subtitle="A primer on mixture of experts, sparse routing, related losses + architectures, and recent models"
                    href="mixture-of-experts-intro"
                />
            </li>
        </ul>
    </div>
}

export default BlogRoot
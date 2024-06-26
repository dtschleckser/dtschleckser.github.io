import { Paper, Text } from '@mantine/core'
import { Link } from 'react-router-dom'
import './BlogListItem.css'

function BlogListItem({ image, title, subtitle, date, href }) {
    return (
        <Link to={href}>
            <Paper shadow="xs" radius="md" p="xl" className='paperItem'>
                <img src={image} style={{ width: 'auto', height: 'auto', maxWidth: '20%', objectFit: 'contain'}} />
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', marginLeft: '1rem', textAlign: 'left' }}>
                    <Text size="xl">
                        {title}
                    </Text>
                    <Text c="dimmed" size="sm" style={{marginBottom: '1rem'}}>
                        {date}
                    </Text>
                    <Text c="dimmed">
                        {subtitle}
                    </Text>

                </div>
            </Paper>
        </Link>
    )
}

export default BlogListItem